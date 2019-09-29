from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import random
import string

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
      id = ''.join([random.choice(string.ascii_letters 
         + string.digits) for n in range(13)]) 
      folder = output_path + '/' + id + '_open/subdir/'
      if not os.path.exists(folder):
        os.makedirs(folder)
        
      f.save(folder + secure_filename(f.filename))
      return render_template('results.html')
		
if __name__ == '__main__':
   app.run(debug = True)