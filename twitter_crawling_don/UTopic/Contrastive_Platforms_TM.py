#!/usr/bin/env python
# coding: utf-8

# In[2]:


args_text = '--base-model sentence-transformers/paraphrase-MiniLM-L6-v2 '+\
            '--dataset all --n-word 2000 --epochs-1 100 --epochs-2 50 ' + \
            '--bsz 32 --stage-2-lr 2e-2 --stage-2-repeat 5 --coeff-1-dist 50 '+ \
            '--n-cluster 20 ' + \
            '--stage-1-ckpt trained_model/news_model_paraphrase-MiniLM-L6-v2_stage1_20t_2000w_99e.ckpt'


# In[3]:


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

#유사도 행렬 저장 경로 설정
#basesim_path = f"./save/{args.dataset}_{bert_name_short}_basesim_matrix_full.pkl"

# if os.path.isfile(basesim_path) == False:
#     model = SentenceTransformer(bert_name.split('/')[-1], device='cuda')
#     base_result_list = []
#     for text in tqdm_notebook(trainds.nonempty_text):
#         base_result_list.append(model.encode(text))
        
#     base_result_embedding = np.stack(base_result_list)
#     basereduced_norm = F.normalize(torch.tensor(base_result_embedding), dim=-1)
#     # 임베딩 간의 내적 계산-> matrix생성 
#     basesim_matrix = torch.mm(basereduced_norm, basereduced_norm.t())
#     ind = np.diag_indices(basesim_matrix.shape[0])
#     # (자기 자신과의 유사도는 -1)
#     basesim_matrix[ind[0], ind[1]] = torch.ones(basesim_matrix.shape[0]) * -1
#     torch.save(basesim_matrix, basesim_path)
# else:
#     basesim_matrix = torch.load(basesim_path)


# In[105]:


# loss 계산 함수 
def dist_match_loss(hidden, alpha=1.0):   #특정 hidden representation과 무작위 가중치 간의 거리에 대한 손실을 계산
    device = hidden.device
    hidden_dim = hidden.shape[-1]
    rand_w = torch.Tensor(np.eye(hidden_dim, dtype=np.float64)).to(device)
    loss_dist_match = get_swd_loss(hidden, rand_w, alpha) 
    # SWD 손실을 계산하여 두 분포간의 차이 측정
    return loss_dist_match

def get_swd_loss(states, rand_w, alpha=1.0):   
    device = states.device
    states_shape = states.shape
    states = torch.matmul(states, rand_w)
    states_t, _ = torch.sort(states.t(), dim=1)
    states_prior = torch.Tensor(np.random.dirichlet([alpha]*states_shape[1], states_shape[0])).to(device) # (bsz, dim)
    states_prior = torch.matmul(states_prior, rand_w) # (dim, dim)
    states_prior_t, _ = torch.sort(states_prior.t(), dim=1) # (dim, bsz)
    return torch.mean(torch.sum((states_prior_t - states_t) ** 2, axis=0))


def _hungarian_match(flat_preds, flat_targets, num_samples, class_num):  
    num_k = class_num
    num_correct = np.zeros((num_k, num_k))
    
    #c1(예측 라벨),c2(실제 라벨)에 대한 일치하는 개수 
    for c1 in range(0, num_k):
        for c2 in range(0, num_k):
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res


# In[106]:


# GPU 메모리 초기화
try:
    del model
except:
    pass
finally:
    torch.cuda.empty_cache()



# In[159]:


temp_basesim_matrix = copy.deepcopy(basesim_matrix)
finetuneds = FinetuneDataset(trainds, temp_basesim_matrix, ratio=1, k=1)
memoryloader = DataLoader(finetuneds, batch_size=bsz * 2, shuffle=False, num_workers=0)
result_list = []


# model.eval()
# with torch.no_grad():
#     for idx, batch in enumerate(tqdm_notebook(memoryloader)):        
#         org_input, _, _, _ = batch
#         org_input_ids = org_input['input_ids'].to(gpu_ids[0])
#         org_attention_mask = org_input['attention_mask'].to(gpu_ids[0])
#         topic, embed = model(org_input_ids, org_attention_mask, return_topic = True)
#         result_list.append(topic)
# result_embedding = torch.cat(result_list)
# _, result_topic = torch.max(result_embedding, 1)


# # Re-formulate the bow

# In[183]:


def dist_match_loss(hiddens, alpha=1.0):
    device = hiddens.device
    hidden_dim = hiddens.shape[-1]
    H = np.random.randn(hidden_dim, hidden_dim)
    Q, R = qr(H) 
    rand_w = torch.Tensor(Q).to(device)
    loss_dist_match = get_swd_loss(hiddens, rand_w, alpha)
    return loss_dist_match

def js_div_loss(hidden1, hidden2):
    m = 0.5 * (hidden1 + hidden2)
    return kldiv(m.log(), hidden1) + kldiv(m.log(), hidden2)


def get_swd_loss(states, rand_w, alpha=1.0):
    device = states.device
    states_shape = states.shape
    states = torch.matmul(states, rand_w)
    states_t, _ = torch.sort(states.t(), dim=1)

    # Random vector with length from normal distribution
    states_prior = torch.Tensor(np.random.dirichlet([alpha]*states_shape[1], states_shape[0])).to(device) # (bsz, dim)
    states_prior = torch.matmul(states_prior, rand_w) # (dim, dim)
    states_prior_t, _ = torch.sort(states_prior.t(), dim=1) # (dim, bsz)
    return torch.mean(torch.sum((states_prior_t - states_t)**2, axis=0))


# In[184]:


class Stage2Dataset(Dataset):
    # 데이터셋 생성
    def __init__(self, encoder, ds, basesim_matrix, word_candidates, k=1, lemmatize=False):
        self.lemmatize = lemmatize
        self.ds = ds
        self.org_list = self.ds.org_list
        self.nonempty_text = self.ds.nonempty_text
        english_stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords_list = set(english_stopwords)
        self.vectorizer = CountVectorizer(vocabulary=word_candidates)
        self.vectorizer.fit(self.preprocess_ctm(self.nonempty_text)) 
        self.bow_list = []
        for sent in tqdm(self.nonempty_text):
            self.bow_list.append(self.vectorize(sent))
            
        sim_weight, sim_indices = basesim_matrix.topk(k=k, dim=-1)
        zip_iterator = zip(np.arange(len(sim_weight)), sim_indices.squeeze().data.numpy())
        self.pos_dict = dict(zip_iterator)
        
        self.embedding_list = []
        encoder_device = next(encoder.parameters()).device
        for org_input in tqdm(self.org_list):
            org_input_ids = org_input['input_ids'].to(encoder_device).reshape(1, -1)
            org_attention_mask = org_input['attention_mask'].to(encoder_device).reshape(1, -1)
            embedding = encoder(input_ids = org_input_ids, attention_mask = org_attention_mask)
            self.embedding_list.append(embedding['pooler_output'].squeeze().detach().cpu())
            
    
    def __len__(self):
        return len(self.org_list)
        
    def preprocess_ctm(self, documents):
        preprocessed_docs_tmp = documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords_list])
                                 for doc in preprocessed_docs_tmp]
        if self.lemmatize:
            lemmatizer = WordNetLemmatizer()
            preprocessed_docs_tmp = [' '.join([lemmatizer.lemmatize(w) for w in doc.split()])
                                     for doc in preprocessed_docs_tmp]
        return preprocessed_docs_tmp
        
    def vectorize(self, text):
        text = self.preprocess_ctm([text])
        vectorized_input = self.vectorizer.transform(text)
        vectorized_input = vectorized_input.toarray().astype(np.float64)
#         vectorized_input = (vectorized_input != 0).astype(np.float64)

        # Get word distribution from BoW
        if vectorized_input.sum() == 0:
            vectorized_input += 1e-8
        vectorized_input = vectorized_input / vectorized_input.sum(axis=1, keepdims=True)
        assert abs(vectorized_input.sum() - vectorized_input.shape[0]) < 0.01
        
        vectorized_label = torch.tensor(vectorized_input, dtype=torch.float)
        return vectorized_label[0]
        
        
    def __getitem__(self, idx):
        pos_idx = self.pos_dict[idx]
        return self.embedding_list[idx], self.embedding_list[pos_idx], self.bow_list[idx], self.bow_list[pos_idx]


# In[37]:


finetuneds = Stage2Dataset(model.encoder, trainds, basesim_matrix, word_candidates, lemmatize=True)    

kldiv = torch.nn.KLDivLoss(reduction='batchmean')
vocab_dict = finetuneds.vectorizer.vocabulary_
vocab_dict_reverse = {i:v for v, i in vocab_dict.items()}
print(n_word)


# # Stage 3

# In[186]:


def measure_hungarian_score(topic_dist, train_target):
    dist = topic_dist
    train_target_filtered = train_target
    flat_predict = torch.tensor(np.argmax(dist, axis=1))
    flat_target = torch.tensor(train_target_filtered).to(flat_predict.device)
    num_samples = flat_predict.shape[0]
    num_classes = dist.shape[1]
    match = _hungarian_match(flat_predict, flat_target, num_samples, num_classes)    
    reordered_preds = torch.zeros(num_samples).to(flat_predict.device)
    for pred_i, target_i in match:
        reordered_preds[flat_predict == pred_i] = int(target_i)
    acc = int((reordered_preds == flat_target.float()).sum()) / float(num_samples)
    return acc


# In[187]:


weight_cands = torch.tensor(weight_cand_matrix.max(axis=1)).cuda(gpu_ids[0]).float()


# # Main

# In[40]:


results_list = []

for i in range(args.stage_2_repeat):
#     # VAE 기반 네트워크 구성
#     model = ContBertTopicExtractorAE(N_topic=n_topic, N_word=args.n_word, bert=bert_name, bert_dim=768)
#     # stage 1에서의 모델을 불러움
#     model.load_state_dict(torch.load(model_stage1_name), strict=True)
#     model.beta = nn.Parameter(torch.Tensor(model.N_topic, n_word))
#     nn.init.xavier_uniform_(model.beta)
#     model.beta_batchnorm = nn.Sequential()
#     model.cuda(gpu_ids[0])
    
    losses = AverageMeter()
    dlosses = AverageMeter() 
    rlosses = AverageMeter()
    closses = AverageMeter()
    distlosses = AverageMeter()
    trainloader = DataLoader(finetuneds, batch_size=bsz, shuffle=True, num_workers=0)
    memoryloader = DataLoader(finetuneds, batch_size=bsz * 2, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.stage_2_lr)

    memory_queue = F.softmax(torch.randn(512, n_topic).cuda(gpu_ids[0]), dim=1)
    print("Coeff   / regul: {:.5f} - recon: {:.5f} - c: {:.5f} - dist: {:.5f} ".format(args.coeff_2_regul, 
                                                                                        args.coeff_2_recon,
                                                                                        args.coeff_2_cons,
                                                                                        args.coeff_2_dist))
    for epoch in range(50):
        model.train()
        model.encoder.eval()
        for batch_idx, batch in enumerate(trainloader):       
            org_input, pos_input, org_bow, pos_bow = batch
            org_input = org_input.cuda(gpu_ids[0])
            org_bow = org_bow.cuda(gpu_ids[0])
            pos_input = pos_input.cuda(gpu_ids[0])
            pos_bow = pos_bow.cuda(gpu_ids[0])

            batch_size = org_input_ids.size(0)

            org_dists, org_topic_logit = model.decode(org_input)
            pos_dists, pos_topic_logit = model.decode(pos_input)

            org_topic = F.softmax(org_topic_logit, dim=1)
            pos_topic = F.softmax(pos_topic_logit, dim=1)

            # reconstruction loss
            # batchmean
#             org_target = torch.matmul(org_topic.detach(), weight_cands)
#             pos_target = torch.matmul(pos_topic.detach(), weight_cands)
            
#             _, org_target = torch.max(org_topic.detach(), 1)
#             _, pos_target = torch.max(pos_topic.detach(), 1)
            
            recons_loss = torch.mean(-torch.sum(torch.log(org_dists + 1E-10) * (org_bow * weight_cands), axis=1), axis=0)
            recons_loss += torch.mean(-torch.sum(torch.log((1-org_dists) + 1E-10) * ((1-org_bow) * weight_cands), axis=1), axis=0)
            recons_loss += torch.mean(-torch.sum(torch.log(pos_dists + 1E-10) * (pos_bow * weight_cands), axis=1), axis=0)
            recons_loss += torch.mean(-torch.sum(torch.log((1-pos_dists) + 1E-10) * ((1-pos_bow) * weight_cands), axis=1), axis=0)
            recons_loss *= 0.5

            # consistency loss
            pos_sim = torch.sum(org_topic * pos_topic, dim=-1)
            cons_loss = -pos_sim.mean()

            # distribution loss
            # batchmean
            distmatch_loss = dist_match_loss(torch.cat((org_topic, pos_topic), dim=0), dirichlet_alpha_2)
            

            loss = args.coeff_2_recon * recons_loss + \
                   args.coeff_2_cons * cons_loss + \
                   args.coeff_2_dist * distmatch_loss 
            
            losses.update(loss.item(), bsz)
            closses.update(cons_loss.item(), bsz)
            rlosses.update(recons_loss.item(), bsz)
            distlosses.update(distmatch_loss.item(), bsz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print("Epoch-{} / recon: {:.5f} - dist: {:.5f} - cons: {:.5f}".format(epoch, rlosses.avg, distlosses.avg, closses.avg))

    print("------- Evaluation results -------")
    all_list = {}
    for e, i in enumerate(model.beta.cpu().topk(10, dim=1).indices):
        word_list = []
        for j in i:
            word_list.append(vocab_dict_reverse[j.item()])
        all_list[e] = word_list
        print("topic-{}".format(e), word_list)

    topic_words_list = list(all_list.values())
    now = datetime.now().strftime('%y%m%d_%H%M%S')
    results = get_topic_qualities(topic_words_list, palmetto_dir=args.palmetto_dir,
                                  reference_corpus=[doc.split() for doc in trainds.preprocess_ctm(trainds.nonempty_text)],
                                  filename=f'results/{now}.txt')
    
    if should_measure_hungarian:
        topic_dist = torch.empty((0, n_topic))
        model.eval()
        evalloader = DataLoader(finetuneds, batch_size=bsz, shuffle=False, num_workers=0)
        non_empty_text_index = [i for i, text in enumerate(textData.data) if len(text) != 0]
        assert len(finetuneds) == len(non_empty_text_index)
        with torch.no_grad():
            for batch in tqdm(evalloader):
                org_input, _, org_bow, __ = batch
                org_input = org_input.cuda(gpu_ids[0])
                org_dists, org_topic_logit = model.decode(org_input)
                org_topic = F.softmax(org_topic_logit, dim=1)
                topic_dist = torch.cat((topic_dist, org_topic.detach().cpu()), 0)
        label_accuracy = measure_hungarian_score(
                             topic_dist,
                             [target for i, target in enumerate(textData.targets)
                              if i in non_empty_text_index]
                         )
        results['label_match'] = label_accuracy

    print(results)
    print()
    results_list.append(results)


# In[189]:


results_df = pd.DataFrame(results_list)
print(results_df)
print('mean')
print(results_df.mean())
print('std')
print(results_df.std())

if args.result_file is not None:
    result_filename = f'results/{args.result_file}'
else:
    result_filename = f'results/{now}.tsv'

results_df.to_csv(result_filename, sep='\t')


# In[190]:


results_df


# In[ ]:


# end of documents

