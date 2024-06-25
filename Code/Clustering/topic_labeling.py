from together import Together
import os
import pandas as pd

TOGETHER_API_KEY = 'XX'
client = Together(api_key=TOGETHER_API_KEY)

# cluster = pd.read_csv('../Data/BERT_topic/clustered_documents.csv')
cluster = pd.read_csv('../Data/BERT_topic/shared_tweets_topics.csv')
# cluster = pd.read_csv('../Data/BERT_topic/shared_tweets_topics_min7.csv')
topic_numbers = set(cluster['topic'])
def get_topic_label(content):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf", # model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role":"system",
                   "content": "Here is a cluster of hate speech tweets. Based on the content of the tweets, can you give the cluster an understandable name that describes its content? Report a selection of 5 names. Only report the names."},
                              {"role": "user", "content": content}],
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

topic_labels = {}
for i in topic_numbers:
    topic = cluster[cluster['topic'] == i]
    content = topic['content'].tolist()
    content = '; '.join(content)
    print("This is topic", i)
    topic_label = get_topic_label(content)
    topic_labels[i] = topic_label

cluster['topic_label'] = cluster['topic'].map(topic_labels)
cluster.to_csv('../Data/BERT_topic/cluster_labels.csv', index=False)
