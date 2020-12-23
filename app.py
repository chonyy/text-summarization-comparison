from flask import Flask, jsonify, request
from flask import render_template
from utils import summarize_NLTK

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.form['text']
    summarization = summarize_NLTK(input_text)
    print('result', summarization)
    return jsonify({'summarization': summarization})

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
