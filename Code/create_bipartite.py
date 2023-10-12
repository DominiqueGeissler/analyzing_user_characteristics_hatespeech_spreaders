import pickle
import pandas as pd
import networkx as nx
import numpy as np
from utility.helper import lda_topic_analysis
from logger import get_logger


logger = get_logger(__name__)

logger.info("Create user-hatespeech bipartite graph...")

"""Create user-hatespeech bipartite graph"""


logger.info("Load data...")
# load user attributes
f = open("../Data/source/user_attribute.pkl", "rb")
users=pickle.load(f)
f.close()

# load root tweets
tweets=pd.read_csv('../Data/source/root_tweets.csv', header=0)
tweets['id'] = tweets['id'].astype(str)
tweets=tweets.set_index('id').T.to_dict('list')

logger.info("Create bipartite graph...")
# filter root tweets for retweeted tweets and create edges(user, tweet)
edges_list=[]
tweets_list=[]
for u in users.keys():
    user_tweets=users[u]['root_tweet_ids']
    for tweet_id in user_tweets:
        if tweet_id not in tweets.keys():
            continue
        # keep only tweets with label = 1 (hate speech)
        elif tweets[tweet_id][0] == 1:
            tweets_list.append(tweet_id)
            edges_list.append((u, tweet_id))

# create bipartite graph and remove isolated nodes
B=nx.Graph()
B.add_nodes_from(users.keys(),bipartite=0)
B.add_nodes_from(tweets_list,bipartite=1)
B.add_edges_from(edges_list)
B.remove_nodes_from(list(nx.isolates(B)))

logger.debug(f"Nr of edges in the graph after removing isolated nodes: {B.number_of_edges()}")
logger.debug(f"Nr of nodes in the graph after removing isolated nodes: {B.number_of_nodes()}")
logger.debug(f"Is the graph connected? {nx.is_connected(B)}")

edges_list=list(B.edges)
user_nodes=[]
tweets_list=[]
for e in edges_list:
    user_nodes.append(e[0])
    tweets_list.append(e[1])

user_nodes=list(set(user_nodes))
tweets_list=list(set(tweets_list))
logger.debug(f"Nr of unique hatespeech tweets: {len(tweets_list)}.")
logger.debug(f"Nr of unique retweeters: {len(user_nodes)}.")
graphs = list(B.subgraph(c) for c in nx.connected_components(B))

logger.info("Compute probability")
# save users and tweets remap ids and user labels
probability={} # prob of a root tweet to be retweeted
tweets_map={}
users_map={}
id_root_tweet=0
id_u=0


f1=open('../Data/Graph/users_list.txt', 'w')
f2=open('../Data/Graph/tweets_list.txt', 'w')
f1.write('org_id remap_id\n')
f2.write('org_id remap_id\n')
f3=open('../Data/Graph/user_label.txt', 'w') # label = number of hatespeech retweets the user did

for (node, val) in B.degree():
    if node in tweets_list:
        tweets_map[node]=id_root_tweet
        f2.write(node +' ' + str(id_root_tweet) + '\n')
        probability[node]=1./val
        id_root_tweet += 1
    elif node in user_nodes:
        users_map[node] = id_u
        f1.write(node + ' ' + str(id_u) + '\n')
        f3.write(node + ' '+str(val)+'\n')
        id_u += 1

f1.close()
f2.close()
f3.close()

# compute the probability for each edge
p_edges=[]
for e in edges_list:
    p_edges.append(probability[e[1]])
sum_p=sum(p_edges)
prob=[p/sum_p for p in p_edges]

# create test set based on edge probability
# remove test edges from graph
N=len(edges_list)
N_test=int(N*0.1)
logger.info("Create test set...")
logger.debug(f"Size of test set: {N_test}.")
test_id=np.random.choice(range(N),N_test,replace=False,p=prob)
test=[edges_list[i] for i in test_id]
test_users_interactions={}
for e in test:
    edges_list.remove(e)
    B.remove_edge(e[0],e[1])
    if users_map[e[0]] not in test_users_interactions.keys():
        test_users_interactions.setdefault(users_map[e[0]],[str(tweets_map[e[1]])])
    else:
        test_users_interactions[users_map[e[0]]].append(str(tweets_map[e[1]]))

# save test set as (user1 tweet1 tweet2 ... \n)
# save test propensity score as (user1 tweet1 \n user1 tweet2 \n ...)

f1=open('../Data/test.txt','w')
f_pscore=open('../Data/test-pscore.txt','w')

for u in test_users_interactions.keys():
    f1.write(str(u)+' '+' '.join(test_users_interactions[u])+'\n')
    for n in test_users_interactions[u]:
        f_pscore.write(str(u)+' '+n+'\n')
f_pscore.close()
f1.close()

logger.info("Create train set...")
logger.info(f"Size of train set: {len(edges_list)}.")
# create train set based on remaining edges
graphs = list(B.subgraph(c) for c in nx.connected_components(B))
train_users_interactions={}
for e in edges_list:
    if users_map[e[0]] not in train_users_interactions.keys():
        train_users_interactions.setdefault(users_map[e[0]], [str(tweets_map[e[1]])])
    else:
        train_users_interactions[users_map[e[0]]].append(str(tweets_map[e[1]]))

# save train set as (user1 tweet1 tweet2 ... \n)
# save train propensity score as (user1 tweet1 \n user1 tweet2 \n ...)

f2 = open('../Data/train.txt', 'w')
f_pscore = open('../Data/train-pscore.txt', 'w')

for u in train_users_interactions.keys():
    f2.write(str(u)+' '+' '.join(train_users_interactions[u])+'\n')
    for n in train_users_interactions[u]:
        f_pscore.write(str(u) + ' ' + n + '\n')
f_pscore.close()
f2.close()

# do topic analysis with LDA
tweets_used = {k: tweets[k][1] for k in tweets_list}
logger.info("Compute LDA topic model...")
lda_topic_analysis(tweets_used)

logger.info("Compute user attributes...")
# save user attributes of graph

f3 = open('../Data/user_attribute.txt', 'w')
key_to_remove = ["root_tweet_ids", "label", 'embedding']
for u in users.keys():
    if u in user_nodes:
        attributes=[]
        keys=users[u]
        for k in keys:
            if k not in key_to_remove:
                values=users[u][k]
                if isinstance(values,int):
                    attributes.append(values)
                else:
                    attributes+=values.tolist()
        attributes = [str(a) for a in attributes]
        f3.write(str(users_map[u])+' '+' '.join(attributes)+'\n')
f3.close()

logger.info("Finished.")