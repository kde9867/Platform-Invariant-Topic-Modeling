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
    def __init__(self, path='twitter_total_preprocessed.csv', sample_size=20000):
        self.df = pd.read_csv(path, lineterminator="\n").dropna(subset=['content'])
        self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)

        self.texts = [str(text) for text in self.df['content'].tolist() if text.strip() != ""]
        self.targets = torch.arange(len(self.texts), dtype=torch.long)
        ## platform label
        self.labels = ['twitter'] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.labels[idx]

class RedditDataset(Dataset):
    def __init__(self, path='reddit_total_preprocessed_cleaned.csv', sample_size=20000):
        self.df = pd.read_csv(path).dropna(subset=['preprocessed_text'])
        self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)

        self.texts = [str(text) for text in self.df['preprocessed_text'].tolist() if text.strip() != ""]
        self.targets = torch.arange(len(self.texts), dtype=torch.long) 
        ## platform label
        self.labels = ['reddit'] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.labels[idx]

class YoutubeDataset(Dataset):
    def __init__(self, path='youtube_preprocessed.csv', sample_size=20000):
        self.df = pd.read_csv(path).dropna(subset=['comment_Text'])
        self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)

        self.texts = [str(text) for text in self.df['comment_Text'].tolist() if text.strip() != ""]
        self.targets = torch.arange(len(self.texts), dtype=torch.long) 
        ## platform label
        self.labels = ['youtube'] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.labels[idx] 
    
class BertDataset(Dataset):
    def __init__(self, bert, text_list, platform_label, N_word, vectorizer=None, lemmatize=False):
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
        
        self.jargons =  set(['coinex', 'announces', 'seos', 'chatgptpowered', 'launches', 'bigdata', 'unveils', 'openaichatgpt', 'tags', 'hn', 'stablediffusion', 'chatgptstyle', 'reportedly', 'mba', 'marketers', 'baidu', 'technews', 'fintech', 'chatgptlike', 'elonmusk', 'notion', 'goog', 'googleai', 'digitalmarketing', 'artificalintelligence', 'rt', 'googl', 'bardai', 'edtech', 'malware', 'wharton', 'agix', 'chatgptplus', 'datascience', 'deeplearning', 'msft', 'weirdness', 'tweets', 'amid', 'aitools', 'cybersecurity', 'airdrop', 'cc', 'valentines', 'startups', 'snapchat', 'generativeai', 'buzzfeed', 'fastestgrowing', 'anthropic', 'maker', 'rival', 'techcrunch', 'aiart', 'nocode', 'invests', 'cybercriminals', 'abstracts', 'nyc', 'webinar', 'retweet', 'educators', 'brilliance', 'rescue', 'daysofcode', 'gm', 'rtechnology', 'linkedin', 'licensing', 'copywriting', 'copywriters', 'contentmarketing', 'revolutionizing', 'technologynews', 'warns', 'metaverse', 'cofounder', 'trending', 'founders', 'aipowered', 'openaichat', 'releases', 'microsofts', 'chinas', 'infosec', 'launching', 'jasper', 'nfts', 'newsletter', 'chatgptgod', 'futureofwork', 'digitaltransformation', 'founder', 'feb', 'buzz', 'rn', 'ux', 'courtesy', 'nick', 'claude',
'remindme', 'giphy', 'gif', 'deleted', 'giphydownsized', 'chadgpt', 'removed', 'patched', 'nerfed', 'yup', 'waitlist', 'refresh', 'sydney', 'mods', 'nsfw', 'characterai', 'screenshot', 'downvoted', 'youcom', 'meth', 'ascii', 'karma', 'hahaha', 'hangman', 'chatopenaicom', 'emojis', 'porn', 'redditor', 'vpn', 'upvotes', 'blah', 'upvote', 'violated', 'yep', 'joking', 'nope', 'offended', 'mod', 'bruh', 'roleplay', 'ops', 'bob', 'dans', 'redditors', 'nerf', 'firefox', 'trolling', 'sarcastic', 'huh', 'turbo', 'troll', 'patch', 'tag', 'url', 'sus', 'erotica', 'chad', 'gotcha', 'basilisk', 'login', 'lmfao', 'temperature', 'poll', 'emoji', 'rick', 'dm', 'jailbreak', 'orange', 'sub', 'quack', 'davinci', 'uh', 'flagged', 'op', 'markdown', 'flair', 'cares', 'refreshing', 'hitler', 'cookies', 'hmm', 'yikes', 'erotic', 'gti', 'paywall', 'elaborate', 'yea', 'ah', 'uncensored', 'rude', 'colour', 'bitch', 'therapy', 'neutered', 'deny', 'chats', 'jailbroken', 'cake', 'dungeon', 'dang',
'zronx', 'tuce', 'jontron', 'levy', 'bishop', 'rook', 'thumbnail', 'quotquot', 'jon', 'linus', 'hrefaboutinvalidzcsafeza', 'beluga', 'vid', 'bhai', 'gemx', 'raid', 'ohio', 'circle', 'subscribed', 'anna', 'stare', 'canva', 'napster', 'shapiro', 'sponsor', 'broker', 'websiteapp', 'manoj', 'subscriber', 'bluewillow', 'alex', 'vids', 'legends', 'ryan', 'shes', 'hackbanzer', 'quotoquot', 'pictory', 'youtuber', 'profitable', 'pawn', 'joma', 'folders', 'lifechanging', 'thomas', 'ur', 'plz', 'mike', 'scott', 'casey', 'adrian', 'enjoyed', 'stockfish', 'invideo', 'shortlisted', 'hikaru', 'bless', 'corpsb', 'chatgbt', 'bfuture', 'curve', 'accent', 'amc', 'tutorials', 'gotham', 'mrs', 'earning', 'bra', 'elo', 'oliver', 'youtubers', 'quotcontinuequot', 'membership', 'labels', 'dagogo', 'eonr', 'hai', 'quotai', 'affiliate', 'congratulationsbryou', 'subscribers', 'thumbnails', 'azn', 'beast', 'tom', 'trader', 'garetz', 'quot', 'subbed', 'pls', 'quotchatgpt', 'gtp', 'machina', 'quoti', 'bret', 'terminator', 'watchingbrdm', 'quothow', 'nowi', 'mint'])

        
        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.model = AutoModel.from_pretrained(bert)
        self.stopwords_list = set(TfidfVectorizer(stop_words="english").get_stop_words()).union(self.jargons)
        self.N_word = N_word
        
        if vectorizer == None:
            self.vectorizer = TfidfVectorizer(stop_words=None, max_features=self.N_word, token_pattern=r'\b[a-zA-Z]{2,}\b')
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
        ## platform label    
        self.platform_label_list = platform_label
            
        
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
    
    # mean_pooling 함수 정의
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    
    def __len__(self):
        return len(self.nonempty_text)
    

    def __getitem__(self, idx):
        sentence = self.nonempty_text[idx]
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # mean_pooling 함수를 사용하여 문장 임베딩의 평균을 계산
        pooled_embedding = self.mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])

        # 임베딩과 attention mask 반환
        return self.org_list[idx], self.bow_list[idx], pooled_embedding, self.platform_label_list[idx]

    
    
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
    def __init__(self, encoder, ds, similarity_matrix, N_word, k=1, lemmatize=False):
        self.lemmatize = lemmatize
        self.ds = ds
        self.org_list = self.ds.org_list
        self.nonempty_text = self.ds.nonempty_text
        self.N_word = N_word
        
        self.jargons = set(['coinex', 'announces', 'seos', 'chatgptpowered', 'launches', 'bigdata', 'unveils', 'openaichatgpt', 'tags', 'hn', 'stablediffusion', 'chatgptstyle', 'reportedly', 'mba', 'marketers', 'baidu', 'technews', 'fintech', 'chatgptlike', 'elonmusk', 'notion', 'goog', 'googleai', 'digitalmarketing', 'artificalintelligence', 'rt', 'googl', 'bardai', 'edtech', 'malware', 'wharton', 'agix', 'chatgptplus', 'datascience', 'deeplearning', 'msft', 'weirdness', 'tweets', 'amid', 'aitools', 'cybersecurity', 'airdrop', 'cc', 'valentines', 'startups', 'snapchat', 'generativeai', 'buzzfeed', 'fastestgrowing', 'anthropic', 'maker', 'rival', 'techcrunch', 'aiart', 'nocode', 'invests', 'cybercriminals', 'abstracts', 'nyc', 'webinar', 'retweet', 'educators', 'brilliance', 'rescue', 'daysofcode', 'gm', 'rtechnology', 'linkedin', 'licensing', 'copywriting', 'copywriters', 'contentmarketing', 'revolutionizing', 'technologynews', 'warns', 'metaverse', 'cofounder', 'trending', 'founders', 'aipowered', 'openaichat', 'releases', 'microsofts', 'chinas', 'infosec', 'launching', 'jasper', 'nfts', 'newsletter', 'chatgptgod', 'futureofwork', 'digitaltransformation', 'founder', 'feb', 'buzz', 'rn', 'ux', 'courtesy', 'nick', 'claude',

'remindme', 'giphy', 'gif', 'deleted', 'giphydownsized', 'chadgpt', 'removed', 'patched', 'nerfed', 'yup', 'waitlist', 'refresh', 'sydney', 'mods', 'nsfw', 'characterai', 'screenshot', 'downvoted', 'youcom', 'meth', 'ascii', 'karma', 'hahaha', 'hangman', 'chatopenaicom', 'emojis', 'porn', 'redditor', 'vpn', 'upvotes', 'blah', 'upvote', 'violated', 'yep', 'joking', 'nope', 'offended', 'mod', 'bruh', 'roleplay', 'ops', 'bob', 'dans', 'redditors', 'nerf', 'firefox', 'trolling', 'sarcastic', 'huh', 'turbo', 'troll', 'patch', 'tag', 'url', 'sus', 'erotica', 'chad', 'gotcha', 'basilisk', 'login', 'lmfao', 'temperature', 'poll', 'emoji', 'rick', 'dm', 'jailbreak', 'orange', 'sub', 'quack', 'davinci', 'uh', 'flagged', 'op', 'markdown', 'flair', 'cares', 'refreshing', 'hitler', 'cookies', 'hmm', 'yikes', 'erotic', 'gti', 'paywall', 'elaborate', 'yea', 'ah', 'uncensored', 'rude', 'colour', 'bitch', 'therapy', 'neutered', 'deny', 'chats', 'jailbroken', 'cake', 'dungeon', 'dang',

'zronx', 'tuce', 'jontron', 'levy', 'bishop', 'rook', 'thumbnail', 'quotquot', 'jon', 'linus', 'hrefaboutinvalidzcsafeza', 'beluga', 'vid', 'bhai', 'gemx', 'raid', 'ohio', 'circle', 'subscribed', 'anna', 'stare', 'canva', 'napster', 'shapiro', 'sponsor', 'broker', 'websiteapp', 'manoj', 'subscriber', 'bluewillow', 'alex', 'vids', 'legends', 'ryan', 'shes', 'hackbanzer', 'quotoquot', 'pictory', 'youtuber', 'profitable', 'pawn', 'joma', 'folders', 'lifechanging', 'thomas', 'ur', 'plz', 'mike', 'scott', 'casey', 'adrian', 'enjoyed', 'stockfish', 'invideo', 'shortlisted', 'hikaru', 'bless', 'corpsb', 'chatgbt', 'bfuture', 'curve', 'accent', 'amc', 'tutorials', 'gotham', 'mrs', 'earning', 'bra', 'elo', 'oliver', 'youtubers', 'quotcontinuequot', 'membership', 'labels', 'dagogo', 'eonr', 'hai', 'quotai', 'affiliate', 'congratulationsbryou', 'subscribers', 'thumbnails', 'azn', 'beast', 'tom', 'trader', 'garetz', 'quot', 'subbed', 'pls', 'quotchatgpt', 'gtp', 'machina', 'quoti', 'bret', 'terminator', 'watchingbrdm', 'quothow', 'nowi', 'mint'
                  ])
        
#         english_stopwords = nltk.corpus.stopwords.words('english')
#         self.stopwords_list = set(english_stopwords)
        self.stopwords_list = set(TfidfVectorizer(stop_words="english").get_stop_words()).union(self.jargons)
        
        self.vectorizer = TfidfVectorizer(stop_words=None, max_features=self.N_word, token_pattern=r'\b[a-zA-Z]{2,}\b')
        self.vectorizer.fit(self.preprocess_ctm(self.nonempty_text)) 
        self.bow_list = []
        for sent in tqdm(self.nonempty_text):
            self.bow_list.append(self.vectorize(sent))
            
#         self.similarity_matrix = torch.tensor(similarity_matrix)  # NumPy 배열을 PyTorch 텐서로 변환
#         sim_weight, sim_indices = self.similarity_matrix.topk(k=k, dim=-1)
#         zip_iterator = zip(np.arange(len(sim_weight)), sim_indices.squeeze().data.numpy())
#         self.pos_dict = dict(zip_iterator)
        self.pos_dict = similarity_matrix
        
        self.embedding_list = []
        encoder_device = next(encoder.parameters()).device
        for org_input in tqdm(self.org_list):
            org_input_ids = org_input['input_ids'].to(encoder_device).reshape(1, -1)
            org_attention_mask = org_input['attention_mask'].to(encoder_device).reshape(1, -1)
            embedding = encoder(input_ids = org_input_ids, attention_mask = org_attention_mask)
            self.embedding_list.append(embedding['pooler_output'].squeeze().detach().cpu())
            
        self.platform_label_list = self.ds.platform_label_list
            
    
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
        return idx, self.embedding_list[idx], self.embedding_list[pos_idx], self.bow_list[idx], self.bow_list[pos_idx],self.platform_label_list[idx]
