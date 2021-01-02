import warnings
from src.CNN.CNN import Seq2Seq, summarize_CNN, Encoder, AttentionDecoder
from src.LSTM_BiLSTM.bilstm import summarize_BiLSTM
from src.LSTM_BiLSTM.lstm import summarize_LSTM
from src.utils import summarize_nltk, summarize_BERT, summarize_T5
from flask import Flask, jsonify, request, escape, render_template, Markup
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize_NLTK_LSTM_BiLSTM', methods=['POST'])
def summarizeNLTK():
    input_text = str(escape(request.form['text']))

    prev_time = time.time()
    summaryNLTK = summarize_nltk(input_text, 'english')
    clean_summary_NLTK = Markup(summaryNLTK).unescape()
    nltk_time = time.time() - prev_time

    prev_time = time.time()
    summaryLSTM = summarize_LSTM(input_text)
    clean_summary_LSTM = Markup(summaryLSTM).unescape()
    lstm_time = time.time() - prev_time

    prev_time = time.time()
    summaryBiLSTM = summarize_BiLSTM(input_text)
    clean_summary_BiLSTM = Markup(summaryBiLSTM).unescape()
    bilstm_time = time.time() - prev_time

    return jsonify({'summaryNLTK': clean_summary_NLTK, 'summaryLSTM': clean_summary_LSTM, 'summaryBiLSTM': clean_summary_BiLSTM,
                    'timeNLTK': str(round(nltk_time, 1)), 'timeLSTM': str(round(lstm_time, 1)), 'timeBiLSTM': str(round(bilstm_time, 1))})


@app.route('/summarizeBERT', methods=['POST'])
def summarizeBERT():
    input_text = str(escape(request.form['text']))

    prev_time = time.time()
    summary = summarize_BERT(input_text)
    clean_summary = Markup(summary).unescape()
    elapsed_time = time.time() - prev_time

    return jsonify({'summary': clean_summary, 'time': str(round(elapsed_time, 1))})


@app.route('/summarizeT5', methods=['POST'])
def summarizeT5():
    input_text = str(escape(request.form['text']))

    prev_time = time.time()
    summary = summarize_T5(input_text)
    clean_summary = Markup(summary).unescape()
    elapsed_time = time.time() - prev_time

    return jsonify({'summary': clean_summary, 'time': str(round(elapsed_time, 1))})


@app.route('/summarizeCNN', methods=['POST'])
def summarizeCNN():
    input_text = str(escape(request.form['text']))

    prev_time = time.time()
    summary = summarize_CNN(input_text)
    clean_summary = Markup(summary).unescape()
    elapsed_time = time.time() - prev_time

    return jsonify({'summary': clean_summary, 'time': str(round(elapsed_time, 1))})


if __name__ == '__main__':
    app.run(debug=True)
