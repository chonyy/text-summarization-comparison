from flask import Flask, jsonify, request, escape, render_template, Markup

from src.utils import summarize_nltk, summarize_BERT
from src.LSTM_BiLSTM.lstm import summarize_LSTM
from src.LSTM_BiLSTM.bilstm import summarize_BiLSTM

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
    input_text = str(escape(request.form['text']))

    summarization = summarize_BERT(input_text)
    clean_summarization = Markup(summarization).unescape()

    return jsonify({'summarization': clean_summarization})


@app.route('/summarizeLSTM', methods=['POST'])
def summarizeLSTM():
    input_text = str(escape(request.form['text']))

    summarization = summarize_LSTM(input_text)
    clean_summarization = Markup(summarization).unescape()

    return jsonify({'summarization': clean_summarization})


@app.route('/summarizeBiLSTM', methods=['POST'])
def summarizeBiLSTM():
    input_text = str(escape(request.form['text']))

    summarization = summarize_BiLSTM(input_text)
    clean_summarization = Markup(summarization).unescape()

    return jsonify({'summarization': clean_summarization})


if __name__ == '__main__':
    app.run(debug=True)
