import pickle
import pandas as pd
from bertopic import BERTopic
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import umap
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

# load root tweets
print("Load data...")
root_tweets=pd.read_csv('../Data/source/root_tweets.csv', header=0)
root_tweets['id'] = root_tweets['id'].astype(str)

# list of tweet ids from txt file
tweet_ids = []
with open('../Data/BERT_topic/Graph/tweets_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        tweet_ids.append(line.split()[0])

# filter root tweets for retweeted tweets based on tweet ids from txt file
# Filter the DataFrame
shared_tweets = root_tweets[root_tweets['id'].isin(tweet_ids)]
# remove links from tweets
shared_tweets['content'] = shared_tweets['content'].str.replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
# remove emojis from tweets
shared_tweets['content'] = shared_tweets['content'].str.replace(r'[^\x00-\x7F]+', '', regex=True)


print("BERTopic analysis...")
# Initialize BERTopic model
vectorizer_model = CountVectorizer(stop_words="english")
# Customize UMAP
umap_model = umap.UMAP(n_neighbors=15, n_components=7, min_dist=0.1, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
kmeans_model = KMeans(n_clusters=4, random_state=42)
bertopic_model = BERTopic(vectorizer_model=vectorizer_model, language="english", min_topic_size=10,
                          umap_model=umap_model, hdbscan_model=hdbscan_model)


# Fit BERTopic model on tweets
topics, _ = bertopic_model.fit_transform(shared_tweets['content'].tolist())
shared_tweets['topic'] = topics
# count the number of tweets in each topic
topic_counts = shared_tweets['topic'].value_counts()
# Get topic info
topic_info = bertopic_model.get_topic_info()

def get_topic_words(topic_model, topic_id):
    topic = topic_model.get_topic(topic_id)
    if topic:
        return ", ".join([word for word, _ in topic])
    return ""

shared_tweets['topic_words'] = shared_tweets['topic'].apply(lambda x: get_topic_words(bertopic_model, x))

# Save the shared tweets with topics to a csv file
shared_tweets.to_csv('../Data/BERT_topic/shared_tweets_topics.csv')
topic_info.to_csv('../Data/BERT_topic/topic_info.csv')
