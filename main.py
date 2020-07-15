#sample video http://mirrors.standaloneinstaller.com/video-sample/small.avi
#https://standaloneinstaller.com/blog/big-list-of-sample-videos-for-testers-124.html
from app import *


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    [
    html.Div([
        html.Div(
            html.Div(id='page-content', className='content'),
            className='content-container'
        ),
    ], className='container-width'),
    dcc.Graph(id="Graph1"),
    dcc.Interval(
            id='interval-component',
            interval=2*1000, # in milliseconds
            n_intervals=0
        ),


    dcc.Location(id='url', refresh=False),

])


page_1_layout=html.Div([html.H2("Object Detection and Counter"),
html.H4("Please Enable Your Webcam"),
html.P("""The object detector will count the duration of the detected object
in the video frame"""),
html.P("""If webcam does not work, feel free to upload a video below"""),
html.Img(src="/video_feed"),
dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Loading(
            id="loading-1",
            type="default",
            children= html.Div(id='output-data-upload')),
    html.Button('Watch Uploaded Video', id='submit-button', n_clicks=0),
    html.Div(id="view_upload",children=[
            html.Iframe(id="imageout",src="/video_upload", width=640, height=480)]),
])

import cv2
from flask import Flask, Response
from helpers import recognition as rc
rec_engine=rc.Recognition_eng()
OCCURENCE_COUNTER={}

class VideoCamera(object):
    def __init__(self):
        try:
            self.video = cv2.VideoCapture(0)
        except:
            self.video=None

    def __del__(self):
        if self.video is not None:
            self.video.release()

    def get_frame(self):
        if self.video is not None:
            global OCCURENCE_COUNTER
            success, image = self.video.read()
            if image is None:
                return str.encode("")
            retjpeg, OCCURENCE_COUNTER=rec_engine.process_stream(image)
            ret, jpeg = cv2.imencode('.jpg', retjpeg) #original
            return jpeg.tobytes()
        return str.encode("")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

class VideoUpload(object):
    def __init__(self):
        if os.path.isfile('./output.avi'):
            self.video = cv2.VideoCapture('./output.avi')
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.loop=True
        else:
            self.video=None

    def __del__(self):
        if self.video is not None:
            self.video.release()
            self.video = cv2.VideoCapture('./output.avi')
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self):
        if self.video is not None:
            frame_counter = 0
            success, image = self.video.read()
            frame_counter += 1
            cv2.waitKey(VIDEO_UPLOAD_PLAYBACK_SPEED)
            if frame_counter == self.video.get(cv2.CAP_PROP_FRAME_COUNT)-2:
                frame_counter = 0 #Or whatever as long as it is the same as next line
                self.video.release()
                self.video = cv2.VideoCapture('./output.avi')
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if success:
                ret, jpeg = cv2.imencode('.jpg', image) #original
                return jpeg.tobytes()
            else:
                if self.loop:
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video = cv2.VideoCapture('./output.avi')
                    success, image = self.video.read()
                    ret, jpeg = cv2.imencode('.jpg', image) #original
                    return jpeg.tobytes()
                return str.encode("")
        else:
            return str.encode("")

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_upload(video):
    while True:
        frame = video.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@server.route('/video_upload')
def video_upload():
    return Response(gen_upload(VideoUpload()),
                mimetype='multipart/x-mixed-replace; boundary=frame')


@app.callback(Output('Graph1','figure'),
[Input('interval-component', 'n_intervals')])
def update_metrics(n):
    sumoccurence=sum(OCCURENCE_COUNTER.values())
    figure={
        'data': [
            {
            'x':list(OCCURENCE_COUNTER),
            'y': [int(OC*100/sumoccurence) for OC in OCCURENCE_COUNTER.values()],
            'type': 'bar', 'name': 'Occurences (%)'
            }
        ],
        'title':"Occurences(%) of Detected Objects in Video",
        'layout': {
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            },
            'xaxis':{
                    'title':'Objects'
                },
            'yaxis':{
                     'title':'Percentage Duration of Occurence(%)'
                }
        }
    }
    return figure


def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)

    return im

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    decoded = b64decode(content)
    file_like_object = BytesIO(decoded)
    with open(os.path.join("./", name), "wb") as fp:
        fp.write(file_like_object.read())


def parse_contents(contents, filename, date):
    global OCCURENCE_COUNTER
    content_type, content_string = contents.split(',')
    try:
        if any(ext in filename.lower() for ext in ['avi','mp4',"mpg","mpeg"]) :
            save_file("video.avi", content_string)
            var,OCCURENCE_COUNTER=rec_engine.process_uploaded_video("video.avi")
            return html.Div([html.H5("Successfully Processed. Click Watch Uploaded Video")
            ])
        else:
            html.Div([
                'Please Upload avi,mp4,mpg,mpeg type file'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            html.H3('There was an error processing this file.')
        ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname, *rest):
    if pathname == '/':
        return page_1_layout


if __name__ == '__main__':
    app.run_server(debug=True)
