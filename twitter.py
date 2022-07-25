# Install Annoconda https://docs.conda.io/en/latest/miniconda.html
# Install and do select add to envirnment PATH option.
# After successful installation. Run following command in command prompt:
# > pip install tweepy pandas textblob tqdm contractions unicodedata emoji itertools
# After this goto file location and type in command. Dont forgot to update the token and hashtag
# > python twitter.py

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tweepy as tw  # To extarct the twitter data
from tqdm import tqdm
import re
from textblob import TextBlob
import contractions
import unicodedata
import emoji
import itertools
from pandas import json_normalize

consumer_key = "<copy from twitter account>"
consumer_secret = "<copy from twitter account>"
final_csv = 'tweet.csv'
num_of_record = 10
search_words = "#plastic -filter:retweets"

def get_tweets():
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    # Collect tweets
    temp = []
    for tweet in tw.Cursor(api.search_tweets,
                q=search_words,
                lang="en"
                ).items(num_of_record):#Return maximum of 10 tweets
        temp.append(tweet._json)
    # Transform to pandas Dataframe
    df = json_normalize(temp)  
    return df

def clean_text(text):
    '''Return clean text'''
    # remove extra space
    text = ' '.join(text.split())
    # replace accented characters é..
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = emoji.demojize(text)# convert symbol based to text
    text = ' '.join(re.sub(r'https?://\S+', '', text).split())# Remove link
    text = re.sub('@[\w]+','',text)# Remove @ user
    text =' '.join(re.sub('[^a-zA-Z0-9 \n\.]', '', text).split())#remove special character à¸£à¸­ plastic plastic à¸šà¸¸à¹‹à¸‡à¹†"
    text = ' '.join(re.sub(r'\s+', " ", text).split())#remove space and split
    text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()#remove single word
    text = re.sub(r'[^\w\s]', '', text)#remove punc
    text =''.join([i for i in text if not i.isdigit()])#remove digit
    text = re.sub(r"[a-zA-Z]",lambda x :x.group(0).lower()#lower case 
            if x.group(0).islower()
            else x.group(0).lower(),text)
    text = contractions.fix(text) # Replace contractions eg. wouldn't, wasn't
    #Fix misspelled words with 2 plus repition of same character like appple to apple
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    return text

def get_tweet_polarity(tweet):
    '''Return tweet polarity.'''
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def get_tweet_sentiment(polarity):
    '''Return sentiment to positive ,negative and neutral base on polarity score'''
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Get raw tweets from twitter
df = get_tweets()
# Drop duplicate columns
df = df.drop_duplicates(subset='text', keep='first')
# Cleaning unwanted text
df['cleaned_text'] = df.apply(lambda row: clean_text(row['text']), axis=1)
# Sentiment polarity score
df['sentiment_polarity'] = df.apply(lambda row: get_tweet_polarity(row['cleaned_text']), axis=1)
# Sentiment
df['sentiment'] = df.apply(lambda row: get_tweet_sentiment(row['sentiment_polarity']), axis=1)
# Save to csv file
df.to_csv(final_csv, index=False)
