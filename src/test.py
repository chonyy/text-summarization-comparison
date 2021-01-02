from utils import summarize_BERT, summarize_nltk, summarize_T5
from LSTM_BiLSTM.lstm import summarize_LSTM
from LSTM_BiLSTM.bilstm import summarize_BiLSTM
from CNN.CNN import Seq2Seq, summarize_CNN, Encoder, AttentionDecoder

with open('./test.txt') as f:
    text = f.read()

summary_BERT = summarize_BERT(text)
summary_nltk = summarize_nltk(text)
summary_LSTM = summarize_LSTM(text)
summary_BiLSTM = summarize_BiLSTM(text)
summary_CNN = summarize_CNN(text)
summary_T5 = summarize_T5(text)

print()
print(summary_BERT)
print()
print(summary_nltk)
print()
print(summary_T5)
print()
print(summary_LSTM)
print()
print(summary_BiLSTM)
print()
print(summary_CNN)
