from flask import Flask, render_template, url_for, request
from text_generation_app import get_preds
import random
import time

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    start = time.time()
    if request.method == 'POST':
        text = request.form['rawtext']
        options = get_preds(text)
        my_prediction = random.choice(options)
    end = time.time()
    final_time = '{:.2f}'.format((end-start))
    return render_template('index.html', prediction=my_prediction, final_time=final_time)


if __name__ == '__main__':
    app.run(debug=True)
