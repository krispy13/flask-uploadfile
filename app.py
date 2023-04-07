from flask import Flask, render_template, request, send_file
import jinja2
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    vector = pickle.load(open('vect.pkl', 'rb'))

    if request.method == 'POST':
        # message = request.form['message']
        df = pd.read_csv(request.form['message'])
        data = df['v2']
        vect = vector.transform(data).toarray()
        my_prediction = model.predict(vect)
        #print(my_prediction)
        df['v3'] = my_prediction
        df.to_csv('data_updated.csv')
    return render_template('home.html',download())


@app.route('/download')
def download():
    path = 'data_updated.csv'
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

