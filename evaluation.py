import os
from datetime import datetime
from itertools import combinations
import gensim.downloader
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

word2vec_google = gensim.downloader.load('word2vec-google-news-300')

def get_topic_qualities(topic_word_list, reference_corpus, dictionary, **kwargs):
    """
    Get topic coherence, similarity, and topic diversity
    arguments
    =========
    topic_word_list: list of list of str
    reference_corpus: list of list of str
    dictionary: gensim.corpora.dictionary.Dictionary
    """
    if 'filename' in kwargs:
        filename = save_topic_top_keywords(topic_word_list, filename=kwargs['filename'])
    else:
        filename = save_topic_top_keywords(topic_word_list, filename=None)
    
    # CoherenceModel 객체 생성 및 점수 계산
    cm_cv = CoherenceModel(topics=topic_word_list, texts=reference_corpus, dictionary=dictionary, coherence='c_v')
    cm_umass = CoherenceModel(topics=topic_word_list, texts=reference_corpus, dictionary=dictionary, coherence='u_mass')
    cm_c_npmi = CoherenceModel(topics=topic_word_list, texts=reference_corpus, dictionary=dictionary, coherence='c_npmi')
    cm_c_uci = CoherenceModel(topics=topic_word_list, texts=reference_corpus, dictionary=dictionary, coherence='c_uci')

    cv_score = cm_cv.get_coherence()
    umass_score = cm_umass.get_coherence()
    c_npmi_score = cm_c_npmi.get_coherence()
    c_uci_score = cm_c_uci.get_coherence()

    # sim_w2v와 diversity는 유지
    sim = get_average_word2vec_similarity(topic_word_list, word2vec_google)
    all_word_set = set()
    all_word_list = []
    for word_list in topic_word_list:
        all_word_set.update(word_list)
        all_word_list += word_list
    diversity = len(all_word_set) / len(all_word_list)

    return {'topic_N': len(topic_word_list),
            'umass': umass_score,
            'c_v': cv_score,
            'c_npmi': c_npmi_score,
            'c_uci': c_uci_score,
            'sim_w2v': sim,
            'diversity': diversity,
            'filename': filename}


def save_topic_top_keywords(top_keywords_list, filename=None):
    # Save the keywords, seperated with spaces
    if filename is None:
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        filename = f"{now}.txt"
    with open(filename, 'w') as f:
        for keywords in top_keywords_list:
            f.write(' '.join(keywords))
            f.write('\n')
    return filename

def read_palmetto_result(result_text):
    # summarize the result from palmetto JAR
    result_lines = result_text.split('\n')
    if 'org.aksw.palmetto.Palmetto' in result_lines[0]:
        result_lines = result_lines[1:]
    val_l = []
    for line in result_lines:
        if line == '':
            continue
        val = line.split('\t')[1]
        val_l.append(float(val))
    print(val_l)
    #print(sum(val_l) / len(val_l))
    return len(val_l)

def get_average_word2vec_similarity(topic_word_list, model):
    similarity_list = []
    missing_word_count = 0
    for topic, word_list in enumerate(topic_word_list):
        word_list_filtered = [word for word in word_list if model.has_index_for(word)]
        missing_word_count += len(word_list) - len(word_list_filtered)
        for word1, word2 in combinations(word_list_filtered, 2):
            similarity = model.similarity(word1, word2)
            similarity_list.append(similarity)
    return sum(similarity_list) / len(similarity_list)