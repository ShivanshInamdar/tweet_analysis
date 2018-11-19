#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:43:30 2017

@author: shivanshinamdar
"""

from collections import deque
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords as sp
import string
import nltk
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def retain_most_used(d, min_times_used):
    c = [key for key, value in d.items() if value < min_times_used]

    for ways_to_tweet in c:
        del d[ways_to_tweet]

    return d


def index_of_date(df, date, first_or_last='first'):
    df['date'], df['time'] = df.created_at.str.split(' ', 1).str
    new_df = df['date']
    if(first_or_last == 'first'):
        return new_df[new_df == date].index.tolist()[-1]
    elif(first_or_last == 'last'):
        return new_df[new_df == date].index.tolist()[0]


def time_to_number(time_string):
    hour = int(time_string[0:2])
    minute = float(time_string[3:5]) / 60.0
    return round(hour + minute)


class timestamps(object):
    def __init__(self, times):
        self.time = times

    def get_time(self):
        return self.time[11:19]


def data(df, source=None):

    tweet_times = []

    count = 0
    # Date when android tweets stopped
    i = index_of_date(df, '03-08-2017', first_or_last='last')
    # Date of announcement of candidacy
    last = index_of_date(df, '06-14-2015', first_or_last='first')
    while(i < last):
        if(source is not None):
            if(df.source.values[i] == ("Twitter for " + source)):
                try:
                    time_from_data = timestamps(df.created_at[i])
                    time = time_to_number(time_from_data.get_time())
                except (TypeError, ValueError):
                    i += 1
                    continue
                if(time == 24):
                    time = 0
                tweet_times.append(time)
                count += 1
        else:
            try:
                time_from_data = timestamps(df.created_at[i])
                time = time_to_number(time_from_data.get_time())
            except (TypeError, ValueError):
                i += 1
                continue
            if(time == 24):
                time = 0
            tweet_times.append(time)
            count += 1
        i += 1

    times_for_each_hour = []

    times_for_each_hour = [(tweet_times.count(j) / count) *
                           100 for j in range(24)]

    return times_for_each_hour


def plot_time_of_day(times_for_each_hour, src=None):
    d = deque(times_for_each_hour)
    d.rotate(-6)

    plt.figure(figsize=(15, 8))
    lbl, = plt.plot(range(0, 24), d, label=src)
    plt.xlabel("Hours of Day")
    plt.ylabel("Percentage of Tweets")
    plt.title("Time of Tweets")

    return lbl


def freq_dist(df, src=None):
    tokenizer = TweetTokenizer()

    tweets = df.text.values
    source = df.source.values

    # print(tweets[1])
    tweet_words = []

    stopwords = set(sp.words('english'))
    # stopwords.update([ 'I', "\"", "/", '’', "'", "A", "The"])
    stopwords.update(['I', "\"", "/", '’', "'", "A", "The", ':/', 'RT',
                      'Android', 'U', 'S', 'We', 'Twitter', '…',
                      '@realDonaldTrump', '🇺', '🇸'])

    # Date when android tweets stopped
    i = index_of_date(df, '12-01-2017', first_or_last='last')

    # Date of announcement of candidacy
    last = index_of_date(df, '06-14-2015', first_or_last='first')

    while(i < last):
        try:
            if(src is not None):
                if(source[i] == "Twitter for " + src):
                    for word in tokenizer.tokenize(tweets[i]):
                        if(word not in stopwords and word not in
                           string.punctuation):
                            tweet_words.append(word)
            else:
                for word in tokenizer.tokenize(tweets[i]):
                        if(word not in stopwords and word not in
                           string.punctuation):
                            tweet_words.append(word)
        except TypeError:
            pass
        i += 1

    fdist = FreqDist()

    pos_tokens = nltk.pos_tag(tweet_words)

    need = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
            'VBP', 'VBZ', 'NN',
            'NNP', 'NNPS', 'NNS'}

    tweet_words_needed = []
    for token, tag in pos_tokens:
        if(tag in need):
            tweet_words_needed.append(token)

    for token in tweet_words_needed:
        fdist[token] += 1

    plt.figure(figsize=(15, 8))
    fdist.plot(30)
    plt.show()


def sentiment_analysis(df, titl, src=None, type_of_vader='compound'):
    sid = SentimentIntensityAnalyzer()

    tweets = df.text.values

    d_sentiment = {}
    d_count = {}

    # Date when android tweets stopped
    i = index_of_date(df, '03-08-2017', first_or_last='last')
    # Date of announcement of candidacy
    last = index_of_date(df, '06-14-2015', first_or_last='first')
    while(i < last):
        if(src is not None):
            if(df.source.values[i] == "Twitter for " + src):
                t = time_to_number(df.created_at[i][11:19])
                if(t == 24):
                    t = 0
                s = sid.polarity_scores(tweets[i])
                if(t in d_sentiment and t in d_count):
                    d_sentiment[t] += s[type_of_vader]
                    d_count[t] += 1
                else:
                    d_sentiment[t] = s[type_of_vader]
                    d_count[t] = 1
        else:
            t = time_to_number(df.created_at[i][11:19])
            if(t == 24):
                t = 0
            s = sid.polarity_scores(tweets[i])
            if(t in d_sentiment and t in d_count):
                d_sentiment[t] += s[type_of_vader]
                d_count[t] += 1
            else:
                d_sentiment[t] = s[type_of_vader]
                d_count[t] = 1
        i += 1

    d_average_sentiment = [d_sentiment[i]/d_count[i] for i in range(0, 24)]

    d = deque(d_average_sentiment)
    d.rotate(-6)
    plt.title(titl)
    plt.figure(figsize=(15, 8))
    plt.plot(range(0, 24), d)


def data_using(get_data):
    def new_f(*args, **kwargs):
        res = get_data(*args, **kwargs)
        return res
    return new_f
