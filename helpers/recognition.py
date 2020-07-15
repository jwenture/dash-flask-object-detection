# cd .\Desktop\Projects\Almex\work\bigquery\Concurrent\
import cv2
import tensorflow as tf

#tf.config.set_visible_devices([], 'GPU')
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
#tf.debugging.set_log_device_placement(True)
import os
import numpy as np
import collections
import six
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from helpers.string_int_label_map_pb2 import *

#from threading import Thread
#from multiprocessing import Queue, Pool
import datetime

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './Object-detection-master/model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './Object-detection-master/model/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

from google.protobuf import text_format


class FPS:
    def __init__(self):
          # store the start time, end time, and total number of frames
          # that were examined between the start and end intervals
          self._start = None
          self._end = None
          self._numFrames = 0

    def start(self):
          # start the timer
          self._start = datetime.datetime.now()
          return self

    def stop(self):
          # stop the timer
          self._end = datetime.datetime.now()

    def update(self):
          # increment the total number of frames examined during the
          # start and end intervals
          self._numFrames += 1

    def elapsed(self):
          # return the total number of seconds between the start and
          # end interval
          return (self._end - self._start).total_seconds()

    def fps(self):
          # compute the (approximate) frames per second
          return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0):
          # initialize the video camera stream and read the first frame
          # from the stream
          self.stream = cv2.VideoCapture(src)
          (self.grabbed, self.frame) = self.stream.read()

          # initialize the variable used to indicate if the thread should
          # be stopped
          self.stopped = False

    def start(self):
          # start the thread to read frames from the video stream
          Thread(target=self.update, args=()).start()
          return self

    def update(self):
          # keep looping infinitely until the thread is stopped
          while True:
                        # if the thread indicator variable is set, stop the thread
                        if self.stopped:
                                return

                        # otherwise, read the next frame from the stream
                        (self.grabbed, self.frame) = self.stream.read()

    def read(self):
          # return the frame most recently read
          return self.grabbed, self.frame

    def stop(self):
          # indicate that the thread should be stopped
          self.stopped = True

    def getWidth(self):
          # Get the width of the frames
          return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def getHeight(self):
          # Get the height of the frames
          return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def getFPS(self):
          # Get the frame rate of the frames
          return int(self.stream.get(cv2.CAP_PROP_FPS))

    def isOpen(self):
          # Get the frame rate of the frames
          return self.stream.isOpened()

    def setFramePosition(self, framePos):
          self.stream.set(cv2.CAP_PROP_POS_FRAMES, framePos)

    def getFramePosition(self):
          return int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

    def getFrameCount(self):
          return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))



def detect_objects(image_np, sess, image_tensor, boxes, scores, classes, num_detections):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes_, scores_, classes_, num_detections_) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    image_np, box_to_display_str_map=visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes_),
        np.squeeze(classes_).astype(np.int32),
        np.squeeze(scores_),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

    return image_np, box_to_display_str_map


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
    """Overlay labeled boxes on an image with formatted scores and label names.
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.
    Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection
    Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    class_names_only=[]
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i] in six.viewkeys(category_index):
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
                    class_names_only.append(display_str)
            if not skip_scores:
                if not display_str:
                    display_str = '{}%'.format(int(100*scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            if not skip_track_ids and track_ids is not None:
                if not display_str:
                    display_str = 'ID {}'.format(track_ids[i])
                else:
                    display_str = '{}: ID {}'.format(display_str, track_ids[i])
            box_to_display_str_map[box].append(display_str)
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            elif track_ids is not None:
                prime_multipler = _get_multiplier_for_color_randomness()
                box_to_color_map[box] = STANDARD_COLORS[
                  (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
            else:
                box_to_color_map[box] = STANDARD_COLORS[
                  classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
              image,
              box_to_instance_masks_map[box],
              color=color
          )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
              image,
              box_to_instance_boundaries_map[box],
              color='red',
              alpha=1.0
          )
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
              image,
              box_to_keypoints_map[box],
              color=color,
              radius=line_thickness / 2,
              use_normalized_coordinates=use_normalized_coordinates)

    return image,class_names_only


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
        """Adds a bounding box to an image (numpy array).

        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.

        Args:
        image: a numpy array with shape [height, width, 3].
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                          (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
          ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
          coordinates as absolute.
        """
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                 thickness, display_str_list,
                                 use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def _validate_label_map(label_map):
    """Checks if a label map is valid.

    Args:
    label_map: StringIntLabelMap to validate.

    Raises:
    ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if (item.id == 0 and item.name != 'background' and
                item.display_name != 'background'):
            raise ValueError('Label map id 0 is reserved for the background label')


def load_labelmap(path):
    """Loads label map proto.

    Args:
    path: path to StringIntLabelMap proto text file.
    Returns:
    a StringIntLabelMapProto
    """
    with tf.compat.v1.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = StringIntLabelMap()
        try:
                text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
                label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)
    return label_map

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    """Given label map proto returns categories list compatible with eval.

    This function converts label map proto and returns a list of dicts, each of
    which  has the following keys:
    'id': (required) an integer id uniquely identifying this category.
    'name': (required) string representing category name
      e.g., 'cat', 'dog', 'pizza'.
    We only allow class into the list if its id-label_id_offset is
    between 0 (inclusive) and max_num_classes (exclusive).
    If there are several items mapping to the same id in the label map,
    we will only keep the first one in the categories list.

    Args:
    label_map: a StringIntLabelMapProto or None.  If None, a default categories
      list is created with max_num_classes categories.
    max_num_classes: maximum number of (consecutive) label indices to include.
    use_display_name: (boolean) choose whether to load 'display_name' field as
      category name.  If False or if the display_name field does not exist, uses
      'name' field as category names instead.

    Returns:
    categories: a list of dictionaries representing all possible categories.
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
                categories.append({
                  'id': class_id + label_id_offset,
                  'name': 'category_{}'.format(class_id + label_id_offset)
                })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info(
              'Ignore item %d since it falls outside of requested '
              'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories

def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.

    Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

    Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def worker(input_q, output_q):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')
                sess = tf.compat.v1.Session(graph=detection_graph)
       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')


        fps = FPS().start()
        while True:
                fps.update()
                frame = input_q.get()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#) # COLOR_RGB2BGR

                #output_q.put(frame_rgb)
                #output_q.put( detect_objects(frame_rgb, sess, detection_graph) )
                output_q.put( cv2.cvtColor(    detect_objects(frame_rgb, sess, image_tensor, boxes, scores, classes, num_detections) , cv2.COLOR_RGB2BGR) )

        fps.stop()
        sess.close()




label_map = load_labelmap(PATH_TO_LABELS)
categories = convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                           use_display_name=True)
category_index = create_category_index(categories)


class Recognition_eng():
    def __init__(self):
        args={"queue_size":30, "num_workers":2, "input_device":0, "output":0, "display":1, "full_screen":0, "num_frames":30     }
        """
        worker=worker()

        #Multiprocessing: Init input and output Queue and pool of workers
        self.input_q = Queue(maxsize=args["queue_size"])
        self.output_q = Queue(maxsize=args["queue_size"])
        self.pool = Pool(args["num_workers"], worker, (self.input_q,self.output_q))
        """
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.occurence_counter={}

    def process_frame(self,frame):
        self.input_q.put(frame)
        output_rgb =self.output_q.get()
        return output_rgb

    def process_stream(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #detect_objects(frame_rgb, sess, image_tensor, boxes, scores, classes, num_detections)
        output_rgb, classes_= detect_objects(frame_rgb, self.sess, self.image_tensor, self.boxes, self.scores, self.classes, self.num_detections)
        for cl in classes_:
            if cl in self.occurence_counter:
                self.occurence_counter[cl]+=1
            else:
                self.occurence_counter[cl]=1
        output_rgb=cv2.cvtColor( output_rgb , cv2.COLOR_RGB2BGR)
        return output_rgb, self.occurence_counter


    def process_uploaded_video(self, video):
        w,h=640,480
        if os.path.isfile('./output.avi'):
            os.remove('./output.avi')
        cap = cv2.VideoCapture('./video.avi')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('./output.avi',fourcc, 20, (w,h))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #detect_objects(frame_rgb, sess, image_tensor, boxes, scores, classes, num_detections)
            output_rgb, classes_= detect_objects(frame_rgb, self.sess, self.image_tensor, self.boxes, self.scores, self.classes, self.num_detections)
            for cl in classes_:
                if cl in self.occurence_counter:
                    self.occurence_counter[cl]+=1
                else:
                    self.occurence_counter[cl]=1
            output_rgb=cv2.cvtColor( output_rgb , cv2.COLOR_RGB2BGR)
            output_rgb = cv2.resize(output_rgb, (w, h))
            out.write(output_rgb)


        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return  output_rgb, self.occurence_counter #cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
