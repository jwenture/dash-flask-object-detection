from flask import Flask,request
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from base64 import b64decode
from io import BytesIO
import os

VIDEO_UPLOAD_PLAYBACK_SPEED=900

server=Flask(__name__)
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/')
app.title = 'Recognition Webservice'
app.config.suppress_callback_exceptions=True

@server.route('/', methods=['GET'])
def hello_world():
    if request.method == 'GET':
        return 'Proceed to Dash Page on 127.0.0.1:3'
