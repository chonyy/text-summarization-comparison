from utils import summarize_BERT, summarize_nltk

text = r'testing string'

summary_BERT = summarize_BERT(text)
summary_nltk = summarize_nltk(text)

print(summary_BERT)
print(summary_nltk)
