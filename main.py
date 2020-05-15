from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import random
import string
import predict
import cropper
import time
import base64

app = Flask(__name__)
output_path = 'temp/'


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/src/<folder>/<img>')
def serve(folder, img):
    app.send_file('src/' + folder + '/' + img)

@app.route('/uploader', methods=['GET', 'POST'])
def file_uploaded():
    if request.method == 'POST':
        f = request.files['file']
        start = time.time()
        img, base64compat = cropper.cropFileStorageObject(f)
        labels = predict.predictFromTensor(img)
    else: raise TypeError
        
    return render_template('results.html', labels=labels, base64=base64.b64encode(base64compat).decode(), time=time.time()-start)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
