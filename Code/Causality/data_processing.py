import pandas as pd
import pickle
import numpy as np
from Code.utility.parser import parse_args

args = parse_args()

model_type = args.model_type
data_path = args.data_path


# loads data
users_map = pd.read_csv(data_path + '/Graph/users_list.txt', sep=' ')
f = open(data_path + "/source/user_attribute.pkl","rb")
users=pd.DataFrame(pickle.load(f)).T
f.close()
users = users.reset_index()
users.rename(columns={'index': 'org_id'}, inplace=True)


# map to ids and filter out users with no embeddings
user_attributes = pd.merge(users, users_map, on='org_id', how='right').drop(columns=['org_id', 'root_tweet_ids'])
user_attributes.rename(columns={'remap_id': 'user_id'}, inplace=True)

# load embeddings
embedding_path = '../results/bprmf/%s_' % (model_type)
user_embeddings = np.load(embedding_path + 'user_embedding.npy')
user_embeddings = pd.DataFrame(user_embeddings).reset_index()
user_embeddings.rename(columns={'index': 'user_id'}, inplace=True)

# merge embeddings with user attributes
df = pd.merge(user_attributes, user_embeddings, on='user_id')
df.columns = df.columns.astype(str)

df[['verified', 'register_time', 'status_count', 'followers_count', 'friend_count']] = \
    df[['verified', 'register_time', 'status_count', 'followers_count', 'friend_count']].astype(int)
df['label'] = df['label'].astype(float)

# save data
path = data_path + '/regression/%s_' % (model_type)
df.to_csv(path + 'regression_data.csv', index=False)
print("Number of users:", len(df))

# train-test split
train = df.groupby('label').apply(lambda x: x.sample(frac=0.8, random_state=200)).reset_index(level=0, drop=True)
train = train.sample(frac=1)
print("Length of train set:", len(train))
test = df.drop(train.index).sample(frac=1)
print("Length of test set:", len(test))

train.to_csv(path + 'regression_train.csv', index=False)
test.to_csv(path + 'regression_test.csv', index=False)
