from summarizer import Summarizer

with open('test.txt', 'r', encoding="utf-8") as f:
    article_text = f.read()

model = Summarizer()
print('Summarizing...')
summary = model(article_text)
print(summary)
print(type(summary))
