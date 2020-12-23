from flask import Flask, jsonify, request, escape, render_template, Markup

from utils import summarize_nltk

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = escape(request.form['text'])

    summarization = summarize_nltk(input_text, 'english')
    clean_summarization = Markup(summarization).unescape()

    return jsonify({'summarization': clean_summarization})


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
