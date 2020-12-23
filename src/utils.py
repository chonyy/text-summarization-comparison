from collections import defaultdict
from summarizer import Summarizer
import heapq
import re
import nltk


def summarize_nltk(article_text, language='english'):
    max_word_count_for_sentence = 100
    summary_sentence_count = 3

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
    summary = model(article_text, num_sentences=3)
    return summary
