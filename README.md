# Dash Flask Object Detection
Runs Dash/Flask for Object Detection using Tensorflow.

# For Docker build
docker build -t objdetection .
docker run --p 8000:8000 objdetection

Notes:
Running on docker might be considerably slower

# Run
pip install -r requirements
python wsgi.py

Go to link http://127.0.0.1:8000  (Runs better on Chrome than Firefox)

#Sample video files
sample video http://mirrors.standaloneinstaller.com/video-sample/small.avi
https://standaloneinstaller.com/blog/big-list-of-sample-videos-for-testers-124.html
