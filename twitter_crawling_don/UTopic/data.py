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

from sklearn.feature_extraction.text import CountVectorizer
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
    def __init__(self, path='text_preprocess.csv', sample_size=100000):
        self.df = pd.read_csv(path, lineterminator="\n")
        self.df = self.df.sample(n=sample_size, random_state=42)
        self.df.dropna(subset=['content'], inplace=True)

        self.texts = self.df['content'].tolist()
        self.targets = torch.arange(len(self.texts), dtype=torch.long)  

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]


class RedditDataset(Dataset):
    def __init__(self, path='reddit_total_preprocessed_cleaned.csv', sample_size=100000):
        self.df = pd.read_csv(path)
        self.df = self.df.sample(n=sample_size, random_state=42)  
        self.df.dropna(subset=['preprocessed_text'], inplace=True)

        self.texts = self.df['preprocessed_text'].tolist()
        self.targets = torch.arange(len(self.texts), dtype=torch.long) 
        
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]


class YoutubeDataset(Dataset):
    def __init__(self, path='testVideoMetaDataResult_Pre.csv', sample_size=100000):
        self.df = pd.read_csv(path)
        self.df = self.df.sample(n=sample_size, random_state=42)  
        self.df.dropna(subset=['comment_Text'], inplace=True)

        self.texts = self.df['comment_Text'].tolist()
        self.targets = torch.arange(len(self.texts), dtype=torch.long) 
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]



class BertDataset(Dataset):
    def __init__(self, bert, text_list, N_word, vectorizer=None, lemmatize=False):
        self.lemmatize = lemmatize
        self.nonempty_text = [text for text in text_list if len(text) > 0]
        
        # Remove new lines
        self.nonempty_text = [re.sub("\n"," ", sent) for sent in self.nonempty_text]
                
        # Remove Emails
        self.nonempty_text = [re.sub('\S*@\S*\s?', '', sent) for sent in self.nonempty_text]
        
        # Remove new line characters
        self.nonempty_text = [re.sub('\s+', ' ', sent) for sent in self.nonempty_text]
        
        # Remove distracting single quotes
        self.nonempty_text = [re.sub("\'", "", sent) for sent in self.nonempty_text]
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.stopwords_list = set(stopwords.words('english'))
        self.N_word = N_word
        
        if vectorizer == None:
            self.vectorizer = CountVectorizer(stop_words=None, max_features=self.N_word, token_pattern=r'\b[a-zA-Z]{2,}\b')
            self.vectorizer.fit(self.preprocess_ctm(self.nonempty_text))
        else:
            self.vectorizer = vectorizer
            
        self.org_list = []
        self.bow_list = []
        for sent in tqdm(self.nonempty_text):
            org_input = self.tokenizer(sent, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            self.org_list.append(org_input)
            self.bow_list.append(self.vectorize(sent))
            
        
    def vectorize(self, text):
        text = self.preprocess_ctm([text])
        vectorized_input = self.vectorizer.transform(text)
        vectorized_input = vectorized_input.toarray()
        vectorized_input = vectorized_input.astype(np.float64)

        # Get word distribution from BoW
        vectorized_input += 1e-8
        vectorized_input = vectorized_input / vectorized_input.sum(axis=1, keepdims=True)
        assert abs(vectorized_input.sum() - vectorized_input.shape[0]) < 0.01
        vectorized_label = torch.tensor(vectorized_input, dtype=torch.float)
        return vectorized_label[0]
        
        
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
    
    
    def __len__(self):
        return len(self.nonempty_text)
    

    def __getitem__(self, idx):
        return self.org_list[idx], self.bow_list[idx]
    
    
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
    def __init__(self, encoder, ds, basesim_matrix, word_candidates, k=1, lemmatize=False):
        self.lemmatize = lemmatize
        self.ds = ds
        self.org_list = self.ds.org_list
        self.nonempty_text = self.ds.nonempty_text
        english_stopwords = stopwords.words('english')
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
        vectorized_input = vectorized_input.toarray()
        vectorized_input = vectorized_input.astype(np.float64)

        # Get word distribution from BoW
        vectorized_input += 1e-8
        vectorized_input = vectorized_input / vectorized_input.sum(axis=1, keepdims=True)
        assert abs(vectorized_input.sum() - vectorized_input.shape[0]) < 0.01
        vectorized_label = torch.tensor(vectorized_input, dtype=torch.float)
        return vectorized_label[0]
        
        
    def __getitem__(self, idx):
        pos_idx = self.pos_dict[idx]
        return self.embedding_list[idx], self.embedding_list[pos_idx], self.bow_list[idx], self.bow_list[pos_idx]