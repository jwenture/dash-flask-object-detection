FROM python:3.7-slim

WORKDIR usr/src/flask_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1
COPY . .
CMD gunicorn -w 2 -b 0.0.0.0:8000 --timeout 60 wsgi:server
