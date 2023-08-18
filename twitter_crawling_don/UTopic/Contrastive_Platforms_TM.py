
args_text = '--base-model sentence-transformers/paraphrase-MiniLM-L6-v2 '+\
            '--dataset all --n-word 2000 --epochs-1 100 --epochs-2 50 ' + \
            '--bsz 32 --stage-2-lr 2e-2 --stage-2-repeat 5 --coeff-1-dist 50 '+ \
            '--n-cluster 20 ' + \
            '--stage-1-ckpt trained_model/news_model_paraphrase-MiniLM-L6-v2_stage1_20t_2000w_99e.ckpt'


import re
import os
import sys
import time
import copy
import math
import argparse
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtools.optim import RangerLars
import gensim.downloader
import itertools

from scipy.stats import ortho_group
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

import numpy as np
from tqdm import tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import AverageMeter
from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from gensim.corpora.dictionary import Dictionary
from pytorch_transformers import *
from sklearn.mixture import GaussianMixture
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from nltk.corpus import stopwords

from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords

from datetime import datetime
from itertools import combinations
import gensim.downloader
from scipy.linalg import qr
from data import *
from data import TwitterDataset, RedditDataset, YoutubeDataset
from model import ContBertTopicExtractorAE
from evaluation import get_topic_qualities
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')




def _parse_args():
    parser = argparse.ArgumentParser(description='Contrastive topic modeling')
    # 각 stage에서의 epochs수 
    parser.add_argument('--epochs-1', default=50, type=int,
                        help='Number of training epochs for Stage 1')   
    parser.add_argument('--epochs-2', default=10, type=int,
                        help='Number of training epochs for Stage 2')
    #각 stage에서의 batch size
    parser.add_argument('--bsz', type=int, default=64,
                        help='Batch size')
    #data set정의 
    parser.add_argument('--dataset', default='twitter', type=str,
                        choices=['twitter', 'reddit', 'youtube','all'],
                        help='Name of the dataset')
    # 클러스터 수와 topic의 수는 20 (k==20)
    parser.add_argument('--n-cluster', default=20, type=int,
                        help='Number of clusters')
    parser.add_argument('--n-topic', type=int,
                        help='Number of topics. If not specified, use same value as --n-cluster')
    # 단어vocabulary는 2000로 setting
    parser.add_argument('--n-word', default=2000, type=int,
                        help='Number of words in vocabulary')
    
    parser.add_argument('--base-model', type=str,
                        help='Name of base model in huggingface library.')
    
    parser.add_argument('--gpus', default=[0,1,2,3], type=int, nargs='+',
                        help='List of GPU numbers to use. Use 0 by default')
   
    parser.add_argument('--coeff-1-sim', default=1.0, type=float,
                        help='Coefficient for NN dot product similarity loss (Phase 1)')
    parser.add_argument('--coeff-1-dist', default=1.0, type=float,
                        help='Coefficient for NN SWD distribution loss (Phase 1)')
    parser.add_argument('--dirichlet-alpha-1', type=float,
                        help='Parameter for Dirichlet distribution (Phase 1). Use 1/n_topic by default.')
    
    parser.add_argument('--stage-1-ckpt', type=str,
                        help='Name of torch checkpoint file Stage 1. If this argument is given, skip Stage 1.')
 
    parser.add_argument('--coeff-2-recon', default=1.0, type=float,
                        help='Coefficient for VAE reconstruction loss (Phase 2)')
    parser.add_argument('--coeff-2-regul', default=1.0, type=float,
                        help='Coefficient for VAE KLD regularization loss (Phase 2)')
    parser.add_argument('--coeff-2-cons', default=1.0, type=float,
                        help='Coefficient for CL consistency loss (Phase 2)')
    parser.add_argument('--coeff-2-dist', default=1.0, type=float,
                        help='Coefficient for CL SWD distribution matching loss (Phase 2)')
    parser.add_argument('--dirichlet-alpha-2', type=float,
                        help='Parameter for Dirichlet distribution (Phase 2). Use same value as dirichlet-alpha-1 by default.')
    parser.add_argument('--stage-2-lr', default=2e-1, type=float,
                        help='Learning rate of phase 2')
    
    parser.add_argument('--stage-2-repeat', default=5, type=int,
                        help='Repetition count of phase 2')
    
    parser.add_argument('--result-file', type=str,
                        help='File name for result summary')
    parser.add_argument('--palmetto-dir', type=str,
                        help='Directory where palmetto JAR and the Wikipedia index are. For evaluation')
    
    
    # Check if the code is run in Jupyter notebook
    is_in_jupyter = False
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            is_in_jupyter = True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            is_in_jupyter = False  # Terminal running IPython
        else:
            is_in_jupyter = False  # Other type (?)
    except NameError:
        is_in_jupyter = False
    
    if is_in_jupyter:
        return parser.parse_args(args=args_text.split())
    else:
        return parser.parse_args()

def data_load(dataset_name, sample_size=100000):
    should_measure_hungarian = False
    textData = []
    
    if dataset_name == 'twitter':
        dataset = TwitterDataset(sample_size=sample_size)
        textData = [(text, "Twitter") for text in dataset.texts if not (isinstance(text, float) and math.isnan(text))]
    elif dataset_name == 'reddit':
        dataset = RedditDataset(sample_size=sample_size)
        textData = [(text, "Reddit") for text in dataset.texts if not (isinstance(text, float) and math.isnan(text))]
    elif dataset_name == 'youtube':
        dataset = YoutubeDataset(sample_size=sample_size)
        textData = [(text, "YouTube") for text in dataset.texts if not (isinstance(text, float) and math.isnan(text))]
    elif dataset_name == 'all':
        twitter_dataset = TwitterDataset(sample_size=sample_size)
        reddit_dataset = RedditDataset(sample_size=sample_size)
        youtube_dataset = YoutubeDataset(sample_size=sample_size)
        
        # filtering NaN values
        textData += [(text, "Twitter") for text in twitter_dataset.texts if not (isinstance(text, float) and math.isnan(text))]
        textData += [(text, "Reddit") for text in reddit_dataset.texts if not (isinstance(text, float) and math.isnan(text))]
        textData += [(text, "YouTube") for text in youtube_dataset.texts if not (isinstance(text, float) and math.isnan(text))]
    else:
        raise ValueError("Invalid dataset name!")
    
    return textData, should_measure_hungarian



args = _parse_args()
bsz = args.bsz
epochs_1 = args.epochs_1
epochs_2 = args.epochs_2

n_cluster = args.n_cluster
n_topic = args.n_topic if (args.n_topic is not None) else n_cluster
args.n_topic = n_topic

textData, should_measure_hungarian = data_load(args.dataset)

ema_alpha = 0.99
n_word = args.n_word
if args.dirichlet_alpha_1 is None:
    dirichlet_alpha_1 = 1 / n_cluster
else:
    dirichlet_alpha_1 = args.dirichlet_alpha_1
if args.dirichlet_alpha_2 is None:
    dirichlet_alpha_2 = dirichlet_alpha_1
else:
    dirichlet_alpha_2 = args.dirichlet_alpha_2
    
bert_name = args.base_model
bert_name_short = bert_name.split('/')[-1]
gpu_ids = args.gpus

skip_stage_1 = (args.stage_1_ckpt is not None)



documents = textData  
documents = [entry[0] for entry in textData]



def create_single_batch_dataloaders(dataset_list, batch_size=64, shuffle=True, num_workers=0):
    single_batch_list = []

    for dataset in dataset_list:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        single_batch_list.append(next(iter(loader)))

    return single_batch_list

def process_text_data(texts, all_texts):
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=False, token_pattern=r'\b([A-Z]+|[a-z]+)\b')
    vectorizer.fit(all_texts)
    
    tf_matrix_all = vectorizer.transform(all_texts).toarray()
    tf_matrix_texts = vectorizer.transform(texts).toarray()
    
    A = len(all_texts)
    f_all = np.count_nonzero(tf_matrix_all, axis=0)
    W_matrix_all = tf_matrix_all * np.log(1 + A / (f_all**2))
    
    f_texts = np.count_nonzero(tf_matrix_texts, axis=0)
    W_matrix_texts = tf_matrix_texts * np.log(1 + A / (f_texts**2))
    
    # platform global words extraction
    all_tfidf_values = np.sum(tf_matrix_all, axis=0)
    top_n = 10000
    stopwords_indices = all_tfidf_values.argsort()[-top_n:][::-1]
    stopwords = set([vectorizer.get_feature_names()[index] for index in stopwords_indices])
    
    # platform jargons extraction
    jargons = set()  
    
    # stop words + jargons
    exclude_words = stopwords.union(jargons)
    
    # Compute BoW matrix
    bow_vectorizer = TfidfVectorizer(vocabulary=vectorizer.get_feature_names(), 
                                     stop_words=exclude_words, 
                                     lowercase=False, 
                                     token_pattern=r'\b([A-Z]+|[a-z]+)\b')
    bow_vectorizer.fit(texts)
    bow_matrix_texts = bow_vectorizer.transform(texts).toarray()
    
    tf_bow = np.sum(bow_matrix_texts, axis=0)
    f_bow = np.count_nonzero(bow_matrix_texts, axis=0)
    
    adjusted_bow_matrix = bow_matrix_texts / (tf_bow * np.log(1 + A / (f_bow**2)))
    
    return adjusted_bow_matrix, W_matrix_all


def compute_bow_batchwise(batch_texts, full_texts):
    bow_matrix_batch, tf_matrix_full = process_text_data(batch_texts, full_texts)  # Corrected this line

    positive_samples = []
    batch_similarity_matrix = np.dot(bow_matrix_batch, tf_matrix_full.T)
    
    for i in range(len(batch_texts)):
        sorted_indices = np.argsort(batch_similarity_matrix[i])  # 오름차순으로 정렬된 인덱스 반환
        neighbor_index = sorted_indices[-2]  # 자기 자신을 제외하고 가장 유사한 이웃 선택
        positive_samples.append((batch_texts[i], full_texts[neighbor_index]))

    return positive_samples

# 각 데이터셋 초기화
twitter_ds = TwitterDataset()
reddit_ds = RedditDataset()
youtube_ds = YoutubeDataset()

# 첫 번째 배치 데이터 리스트 생성
first_batches = create_single_batch_dataloaders([twitter_ds, reddit_ds, youtube_ds], batch_size=64)

platform_names = ["Twitter", "Reddit", "YouTube"]

for idx, batch in enumerate(first_batches):
    texts, targets = batch
    print(f"Platform: {platform_names[idx]}")
    print("Sample text:", texts[0])
    print("Sample Index:", targets[0])
    print('-' * 50)
    
    # BoW 매트릭스 생성 및 크기 출력
    bow_matrix_batch, tf_matrix_full = process_text_data(texts, twitter_ds.texts)
    print(f"{platform_names[idx]} BoW matrix batch size: {bow_matrix_batch.shape}")
    print(f"{platform_names[idx]} BoW matrix full size: {tf_matrix_full.shape}")
    print('-' * 50)
    
    positive_samples = compute_bow_batchwise(texts, twitter_ds.texts)
    print("Positive sample for first text in batch:", positive_samples[0])
    print('-' * 50)


    
#trainds = BertDataset(bert=bert_name,text_list=textData.texts, N_word=n_word, vectorizer=None, lemmatize=True)

all_data_trainds = BertDataset(bert=bert_name, text_list=textData.texts, N_word=n_word, vectorizer=None, lemmatize=True)
twiiter_trainds = BertDataset(bert=bert_name, text_list=twiiter_data, N_word=n_word, vectorizer=None, lemmatize=True)
reddit_trainds = BertDataset(bert=bert_name, text_list=reddit_data, N_word=n_word, vectorizer=None, lemmatize=True)
youtube_trainds = BertDataset(bert=bert_name, text_list=youtube_data, N_word=n_word, vectorizer=None, lemmatize=True)

# trainds_list에 모든 trainds 인스턴스를 저장
trainds_list = [twitter_trainds, reddit_trainds, youtube_trainds]
