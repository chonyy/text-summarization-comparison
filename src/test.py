from utils import summarize_BERT, summarize_nltk
from LSTM_BiLSTM.lstm import summarize_LSTM
from LSTM_BiLSTM.bilstm import summarize_BiLSTM

with open('./test.txt') as f:
    text = f.read()

summary_BERT = summarize_BERT(text)
summary_nltk = summarize_nltk(text)
summary_LSTM = summarize_LSTM(text)
summary_BiLSTM = summarize_BiLSTM(text)

print()
print(summary_BERT)
print()
print(summary_nltk)
print()
print(summary_LSTM)
print()
print(summary_BiLSTM)
