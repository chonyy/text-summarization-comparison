from flask import Flask, jsonify, request, escape, render_template, Markup

from utils import summarize_nltk, summarize_BERT

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarizeNLTK', methods=['POST'])
def summarizeNLTK():
    input_text = str(escape(request.form['text']))

    summarization = summarize_nltk(input_text, 'english')
    clean_summarization = Markup(summarization).unescape()

    return jsonify({'summarization': clean_summarization})


@app.route('/summarizeBERT', methods=['POST'])
def summarizeBERT():
    input_text = request.form['text']

    summarization = summarize_BERT(input_text)
    clean_summarization = Markup(summarization).unescape()

    return jsonify({'summarization': clean_summarization})


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
