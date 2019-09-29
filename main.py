from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import random
import string
from SE_code.main import main as run
import time

app = Flask(__name__)
output_path = 'src/uploads'


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/src/<folder>/subdir/<img>')
def serve(folder, img):
    app.send_file('src/' + folder + '/subdir/' + img)

@app.route('/uploader', methods=['GET', 'POST'])
def file_uploaded():
    if request.method == 'POST':
        f = request.files['file']
        id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(13)])
        
        folder_ = output_path + '/' + id
        folder = folder_ + '/subdir/'
        # print(os.listdir())
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("made folder: " + folder)

        f.save(folder + secure_filename(f.filename))
        model = "/src/model/20170511-185253.pb"
        classifiers = {
            "id": "/src/classifiers/50-id-classifier.pkl",
            "gen": "/src/classifiers/25-gender-classifier.pkl",
            "age": "/src/classifiers/5-age-classifier.pkl"
        }
        start = time.time()
        # input_directory, model_path, id_classifier_output_path, gen_classifier_output_path, age_classifier_output_path, batch_size, num_threads, is_yes, eval_only)
        labels, path = run(folder_, model, classifiers['id'], classifiers['gen'], classifiers['age'], 128, 1, False, False)
        labels[0]['image_path'] = labels[0]['image_path'][4:]
        
    return render_template('results.html', labels=labels[0], time=time.time()-start)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
