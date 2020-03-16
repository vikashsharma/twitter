import os
import tweepy
import pandas as pd
import re
from textblob import TextBlob 
from bs4 import BeautifulSoup
from pandas import json_normalize
from nltk.tokenize import WordPunctTokenizer

consumer_key = "<copy from twitter account>"
consumer_secret = "<copy from twitter account>"
oauth_token = "<copy from twitter account>"
oauth_token_secret = "<copy from twitter account>"
hash_tag = "plastic"
num_of_record = 10
file_path = 'sample.csv'
search_list = ['plastic','bomb']


tok = WordPunctTokenizer()    
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def get_data(hashtag,limit):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(oauth_token, oauth_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    temp = []
    for tweet in tweepy.Cursor(api.search,q=hashtag).items(limit):
        temp.append(tweet._json)
    df = json_normalize(temp)
    return df

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

def get_tweet_polarity(tweet): 
    analysis = TextBlob(tweet) 
    return analysis.sentiment.polarity

def get_tweet_sentiment(polarity): 
    if polarity > 0: 
        return 'positive'
    elif polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'

def find_words(s):
    tags = []
    for each in search_list:
        if re.findall(r'{}\s+'.format(each),s,re.I):
            tags.append(each)
    return ', '.join(tags)

dataset = get_data(hash_tag,num_of_record)
dataset['cleaned_text']       = dataset.apply(lambda row :tweet_cleaner(row['text']),axis=1)
dataset['sentiment_polarity'] = dataset.apply(lambda row :get_tweet_polarity(row['cleaned_text']),axis=1)
dataset['sentiment']          = dataset.apply(lambda row :get_tweet_sentiment(row['sentiment_polarity']),axis=1)
dataset['labels']             = dataset.apply(lambda row :find_words(row['cleaned_text']),axis=1)
dataset.to_csv(file_path, index = False)
