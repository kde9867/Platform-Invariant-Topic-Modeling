import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora.dictionary import Dictionary

# 단어 빈도 및 단어 쌍 빈도 계산 함수
def calculate_word_frequencies(texts):
    word_freq = defaultdict(int)
    word_pair_freq = defaultdict(int)
    for doc in texts:
        unique_words = set(doc)
        for word in unique_words:
            word_freq[word] += 1
        for word1 in unique_words:
            for word2 in unique_words:
                if word1 != word2:
                    word_pair_freq[(word1, word2)] += 1
    return word_freq, word_pair_freq

# NPMI 점수 계산 함수
def calculate_npmi(topics, word_freq, word_pair_freq, num_docs, epsilon=1e-12):
    npmi_scores_all_topics = []
    for topic_words in topics:
        npmi_scores = []
        for i, word1 in enumerate(topic_words):
            for word2 in topic_words[i+1:]:
                word1_freq = word_freq[word1]
                word2_freq = word_freq[word2]
                word_pair_count = word_pair_freq.get((word1, word2), 0)
                p_word1 = word1_freq / num_docs
                p_word2 = word2_freq / num_docs
                p_word1_word2 = word_pair_count / num_docs
                if p_word1_word2 > 0:
                    npmi = np.log((p_word1_word2 + epsilon) / (p_word1 * p_word2 + epsilon)) / -np.log(p_word1_word2 + epsilon)
                    if np.isfinite(npmi): 
                        npmi_scores.append(npmi)
        if npmi_scores:  # 유효한 점수가 있는 경우에만 평균을 추가
            npmi_scores_all_topics.append(np.mean(npmi_scores))
    return np.mean(npmi_scores_all_topics) if npmi_scores_all_topics else float('nan')

# UCI 점수 계산 함수
def calculate_uci(topics, word_freq, word_pair_freq, num_docs):
    uci_scores_all_topics = []
    for topic_words in topics:
        uci_scores = []
        for i, word1 in enumerate(topic_words):
            for word2 in topic_words[i+1:]:
                word1_freq = word_freq[word1]
                word2_freq = word_freq[word2]
                word_pair_count = word_pair_freq.get((word1, word2), 0)
                p_word1 = word1_freq / num_docs
                p_word2 = word2_freq / num_docs
                p_word1_word2 = word_pair_count / num_docs
                if p_word1_word2 > 0:
                    uci = np.log(p_word1_word2 / (p_word1 * p_word2))
                    uci_scores.append(uci)
        if uci_scores:  # 유효한 점수가 있는 경우에만 평균을 추가
            uci_scores_all_topics.append(np.mean(uci_scores))
    return np.mean(uci_scores_all_topics) if uci_scores_all_topics else float('nan')

# UMass 점수 계산 함수
def calculate_umass(topics, word_freq, word_pair_freq, num_docs):
    umass_scores_all_topics = []
    for topic_words in topics:
        umass_scores = []
        for i, word1 in enumerate(topic_words):
            for word2 in topic_words[i+1:]:
                word1_freq = word_freq[word1]
                word_pair_count = word_pair_freq.get((word1, word2), 0)
                if word_pair_count > 0:
                    umass = np.log((word_pair_count / num_docs) / (word1_freq / num_docs))
                    umass_scores.append(umass)
        if umass_scores:  # 유효한 점수가 있는 경우에만 평균을 추가
            umass_scores_all_topics.append(np.mean(umass_scores))
    return np.mean(umass_scores_all_topics) if umass_scores_all_topics else float('nan')

# CV 점수 계산
def calculate_cv(topics, word_freq, word_pair_freq, num_docs, epsilon=1e-12):
    cv_scores = []
    for topic in topics:
        topic_vector = np.zeros(len(topic))
        word_vectors = []
        for i, word1 in enumerate(topic):
            word_vector = np.zeros(len(topic))
            for j, word2 in enumerate(topic):
                if word1 != word2:
                    word_pair_count = word_pair_freq.get((word1, word2), 0) + epsilon
                    word1_freq = word_freq.get(word1, 0) + epsilon
                    word2_freq = word_freq.get(word2, 0) + epsilon
                    npmi = np.log((word_pair_count / num_docs) / ((word1_freq / num_docs) * (word2_freq / num_docs)))
                    if np.isfinite(npmi):
                        word_vector[j] = npmi
            if np.linalg.norm(word_vector) != 0:
                word_vector /= np.linalg.norm(word_vector)
            topic_vector += word_vector
            word_vectors.append(word_vector)
        if np.linalg.norm(topic_vector) != 0:
            topic_vector /= np.linalg.norm(topic_vector)
        topic_vector_2d = topic_vector.reshape(1, -1)
        cv_scores.append(np.mean([cosine_similarity(topic_vector_2d, wv.reshape(1, -1))[0][0] for wv in word_vectors]))
    return np.mean(cv_scores)

# 주제 다양성 계산 함수
def calculate_topic_diversity(topic_words_list):
    all_words = set()
    for words in topic_words_list:
        all_words.update(words)
    unique_words_count = len(all_words)
    total_words_count = sum(len(words) for words in topic_words_list)
    return unique_words_count / total_words_count

def get_topic_coherence(topics, reference_corpus):
    topics = topics
    texts = reference_corpus
    dictionary = Dictionary(texts).add_documents(topics)
    num_docs = len (texts)
    word_freq, word_pair_freq = calculate_word_frequencies(texts)
    npmi = calculate_npmi(topics, word_freq, word_pair_freq, num_docs, epsilon=1e-12)
    uci = calculate_uci(topics, word_freq, word_pair_freq, num_docs)
    umass = calculate_umass(topics, word_freq, word_pair_freq, num_docs)
    cv = calculate_cv(topics, word_freq, word_pair_freq, num_docs, epsilon=1e-12)
    topic_diversity = calculate_topic_diversity(topics)
    
    return {'NPMI': npmi,
           'UCI' : uci,
           'UMASS' : umass,
           'CV' : cv,
           'Topic_Diversity' : topic_diversity}
