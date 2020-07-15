import os
import requests
from shutil import copyfile

file1="./helpers/model/frozen_inference_graph.pb"
file1_0="./Object-detection-master/model/frozen_inference_graph.pb"
file2="./helpers/model/mscoco_label_map.pbtxt"
file2_0="./Object-detection-master/model/mscoco_label_map.pbtxt"

def download_files():
    if not os.path.isfile(file1):
        r = requests.get("https://github.com/datitran/object_detector_app/raw/master/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb",allow_redirects=True)
        open(file1, 'wb').write(r.content)


    if not os.path.isfile(file2):
        r = requests.get("https://github.com/tensorflow/models/raw/master/research/object_detection/data/mscoco_label_map.pbtxt",allow_redirects=True)
        open(file2, 'wb').write(r.content)

while True:
    if os.path.isfile(file1) and os.path.getsize(file1)<100000:
        os.remove(file1)
        download_files()
    elif not os.path.isfile(file1):
        download_files()
    else:
        break
while True:
    if os.path.isfile(file2) and os.path.getsize(file2)<1000:
        os.remove(file2)
        download_files()
    elif not os.path.isfile(file2):
        download_files()
    else:
        break

if not os.path.isfile(file1_0):
    copyfile(file1, file1_0)

if not os.path.isfile(file2_0):
    copyfile(file2, file2_0)

from main import server

if __name__ == "__main__":
    server.run(host='0.0.0.0', port=8000)
