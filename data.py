import re
import os
import sys
import time
import copy
import glob
import math
import argparse
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader
import itertools

import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from tqdm import tqdm
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from datetime import datetime
from itertools import combinations

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore")


class TwitterDataset(Dataset):
    def __init__(self, path='text_preprocess.csv', sample_size=1000):
        self.df = pd.read_csv(path, lineterminator="\n").dropna(subset=['content'])
        self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        self.platform_label = 'Twitter'

        self.texts = [str(text) for text in self.df['content'].tolist() if text.strip() != ""]
        self.targets = torch.arange(len(self.texts), dtype=torch.long)  

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.platform_label

class RedditDataset(Dataset):
    def __init__(self, path='reddit_total_preprocessed_cleaned.csv', sample_size=1000):
        self.df = pd.read_csv(path).dropna(subset=['preprocessed_text'])
        self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        self.platform_label = 'Reddit'

        self.texts = [str(text) for text in self.df['preprocessed_text'].tolist() if text.strip() != ""]
        self.targets = torch.arange(len(self.texts), dtype=torch.long) 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.platform_label

class YoutubeDataset(Dataset):
    def __init__(self, path='testVideoMetaDataResult_Pre.csv', sample_size=1000):
        self.df = pd.read_csv(path).dropna(subset=['comment_Text'])
        self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        self.platform_label = 'Youtube'

        self.texts = [str(text) for text in self.df['comment_Text'].tolist() if text.strip() != ""]
        self.targets = torch.arange(len(self.texts), dtype=torch.long) 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.platform_label
        

class BertDataset(Dataset):
    def __init__(self, bert, text_list, platform_list, total_bow_matrix, all_text_list, lemmatize=False):
        self.lemmatize = lemmatize
        self.org_text = [text for text in text_list if len(text) > 0]
        self.nonempty_text = [text for text in text_list if len(text) > 0]
        
        self.text_list = text_list  # 텍스트 정보 저장
        self.platform_list = platform_list  # 플랫폼 정보 저장
        
        # Remove new lines
        self.nonempty_text = [re.sub("\n"," ", sent) for sent in self.nonempty_text]
                
        # Remove Emails
        self.nonempty_text = [re.sub('\S*@\S*\s?', '', sent) for sent in self.nonempty_text]
        
        # Remove new line characters
        self.nonempty_text = [re.sub('\s+', ' ', sent) for sent in self.nonempty_text]
        
        # Remove distracting single quotes
        self.nonempty_text = [re.sub("\'", "", sent) for sent in self.nonempty_text]
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert)

        self.org_list = []  # 원본 텍스트 리스트 (이 경우 nonempty_text와 동일)
        self.bow_list = []  # BoW 리스트

        for sent in tqdm(self.nonempty_text):
            org_input = self.tokenizer(sent, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            self.org_list.append(org_input) # org_list에 추가

        # 비어 있지 않은 텍스트에 대한 원본 텍스트 인덱스와 BoW를 찾기
        for text in self.org_text:
            if text in all_text_list:  # text가 all_text_list에 있다면
                org_text_index = all_text_list.index(text)  # 해당 인덱스 찾기
                self.bow_list.append(total_bow_matrix[org_text_index])  # 해당 BoW를 bow_list에 추가
    
    stopwords_list = []
    def preprocess_ctm(self, documents):
        
        stopwords_list = TfidfVectorizer(stop_words="english").get_stop_words()
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
    
    def __len__(self):
        return len(self.nonempty_text)
    

    def __getitem__(self, idx):
        return self.text_list[idx], self.platform_list[idx], self.bow_list[idx]

    
    
class FinetuneDataset(Dataset):
    def __init__(self, ds, basesim_matrix, ratio, k=1):
        self.ds = ds
        self.org_list = self.ds.org_list
        self.bow_list = self.ds.bow_list
        sim_weight, sim_indices = basesim_matrix.topk(k=k, dim=-1)
        zip_iterator = zip(np.arange(len(sim_weight)), sim_indices.squeeze().data.numpy())
        self.pos_dict = dict(zip_iterator)
        
    def __len__(self):
        return len(self.org_list)
        
    def __getitem__(self, idx):
        if idx in self.pos_dict:
            pos_idx = self.pos_dict[idx]
            return self.org_list[idx], self.org_list[pos_idx], self.bow_list[idx], self.bow_list[pos_idx]
        else:
            return self.org_list[idx], self.org_list[idx], self.bow_list[idx], self.bow_list[idx]
        
        
class FinetuneSimCSEDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.org_list = self.ds.org_list
        self.bow_list = self.ds.bow_list
        
    def __len__(self):
        return len(self.org_list)
        
    def __getitem__(self, idx):
        return self.org_list[idx], self.org_list[idx], self.bow_list[idx], self.bow_list[idx]

        
class Stage2Dataset(Dataset):
    def __init__(self, encoder, ds, total_bow_matrix, total_text_list, lemmatize=False):
        self.lemmatize = lemmatize
        self.ds = ds
        self.org_list = self.ds.org_list
        self.nonempty_text = self.ds.nonempty_text
        self.bow_list = self.ds.bow_list
        self.org_text = self.ds.org_text
            
        #여기에 인접한 pos계산을 하고 idx를 저장하는 리스트를 생성
        self.pos_idx_list = []

        # toal_bow_matrix와 bow_matrix 간의 코사인 유사도 계산
        cosine_sim_matrix = cosine_similarity(self.bow_list, total_bow_matrix)
        # batch_texts에 있는 각 텍스트에 대해 full_texts에서 인덱스를 찾음
        for i, batch_text in enumerate(self.org_text):
            # full_texts에서 batch_text의 인덱스 찾기
            self_index = total_text_list.index(batch_text) if batch_text in total_text_list else None
            
            # 찾은 인덱스의 유사도를 -1로 설정
            if self_index is not None:
                cosine_sim_matrix[i][self_index] = -1

        # 각 batch_text에 대해 가장 유사도가 높은 full_text의 인덱스를 찾음
        for i in range(len(self.bow_list)):
            positive_sample_index = np.argmax(cosine_sim_matrix[i])
            self.pos_idx_list.append(positive_sample_index)
        
        
        self.embedding_list = []
        encoder_device = next(encoder.parameters()).device
        for org_input in tqdm(self.org_list):
            org_input_ids = org_input['input_ids'].to(encoder_device).reshape(1, -1)
            org_attention_mask = org_input['attention_mask'].to(encoder_device).reshape(1, -1)
            embedding = encoder(input_ids = org_input_ids, attention_mask = org_attention_mask)
            self.embedding_list.append(embedding['pooler_output'].squeeze().detach().cpu())
        
        self.pos_embedding_list = []
        pos_bow_list = []
        for idx in tqdm(self.pos_idx_list):
            text_data = total_text_list[idx]
        

            pos_input = self.tokenizer(text_data, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            pos_input['input_ids'] = torch.squeeze(pos_input['input_ids'])
            pos_input['attention_mask'] = torch.squeeze(pos_input['attention_mask'])
            
            pos_input_ids = pos_input['input_ids'].to(encoder_device).reshape(1, -1)
            pos_attention_mask = pos_input['attention_mask'].to(encoder_device).reshape(1, -1)
            embedding = encoder(input_ids = pos_input_ids, attention_mask = pos_attention_mask)
            self.pos_embedding_list.append(embedding['pooler_output'].squeeze().detach().cpu())
        
            pos_bow_list.append(total_bow_matrix[idx])

    
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
        
        
    def __getitem__(self, idx):
        return self.embedding_list[idx], self.pos_embedding_list[idx], self.bow_list[idx], self.pos_bow_list[idx]
    
    #여기서 전체 데이터셋에 대한 bow 매트릭스 계산