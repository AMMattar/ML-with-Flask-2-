from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('ml.pkl','rb'))

@app.route('/', methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        sales = [[int(x) for x in request.form.values()]]
        answer = model.predict(sales)
        return render_template('answer.html', answer = answer)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
