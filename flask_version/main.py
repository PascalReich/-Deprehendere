from flask import Flask, render_template, request
from werkzeug import secure_filename
import os

app = Flask(__name__)
output_path = 'uploads'

@app.route('/')
def main():
   return render_template('index.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def file_uploaded():
   if request.method == 'POST':
      f = request.files['file']
      if not os.path.exists(output_path):
        os.makedirs(output_path)

      f.save(output_path + os.sep + secure_filename(f.filename))
      return render_template('results.html')
		
if __name__ == '__main__':
   app.run(debug = True)