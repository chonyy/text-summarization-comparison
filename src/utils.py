from collections import defaultdict
from summarizer import Summarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import json
import heapq
import re
import nltk


def summarize_nltk(article_text, language='english'):
    max_word_count_for_sentence = 50
    summary_sentence_count = 2

    nltk.data.path.append('./nltk_data/')
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)

    stopwords = nltk.corpus.stopwords.words(language)

    word_frequencies = defaultdict(float)
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    sentence_scores = defaultdict(int)
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence.lower()):
            if len(sentence.split()) < max_word_count_for_sentence:
                sentence_scores[sentence] += word_frequencies[word]

    summary_sentences = heapq.nlargest(
        summary_sentence_count, sentence_scores, key=sentence_scores.get)

    indexed_summary_sentences = [(sentence_list.index(
        sentence), sentence) for sentence in summary_sentences]
    aligned_summary_sentences = sorted(
        indexed_summary_sentences, key=lambda x: x[0], reverse=False)
    final_summary_sentences = map(lambda x: x[1], aligned_summary_sentences)

    summary = ' '.join(final_summary_sentences)

    return summary


def summarize_BERT(article_text):
    print('Loading BERT model...')
    model = Summarizer()
    print('Summarizing...')
    summary = model(article_text, num_sentences=2)
    return summary


def summarize_T5(text):
    device = torch.device('cpu')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
