# Causal Understanding of Hate Speech Resharing

This is the code for the paper Causal Understanding of Why Users Share Hate Speech
on Social Media.

The code has been tested on Python 3.8

## Abstract
Hate speech on social media threatens the mental and physical well-being of individuals and is further responsible for real-world violence. An important driver behind the spread of hate speech and thus why hateful posts can go viral are reshares, yet little is known about why users reshare hate speech. In this paper, we present a comprehensive, causal analysis of the user attributes that make users reshare hate speech. However, causal inference from observational social media data is challenging, because the data likely suffers from selection bias, and there is further confounding due to the vulnerability of users to hate speech. We develop a novel, three-step causal framework: (1) We debias our observational social media data by applying inverse propensity scoring. (2) We use the debiased propensity scores to model the latent vulnerability of users to hate speech as a latent embedding, which acts as a confounder in our causal framework. (3) We model the causal effects of user attributes on users’ probability of sharing hate speech. Compared to existing baselines, a particular strength of our framework is that it offers causal inferences that are non-linear, yet still explainable. We find that users with fewer followers, fewer friends, and fewer posts share more hate speech. Younger accounts, in return, share less hate speech. Overall, understanding the factors that drive users to share hate speech is crucial for detecting individuals at risk of engaging in harmful behavior and for designing effective mitigation strategies.

## Repository Structure
Code: Folder to store the code used for the analysis.\
Data: Folder to store the data.
- source: store user attributes and root tweets there to get started.

## Data
We anonymize the usernames by replacing them with dummy usernames. User attributes can be crawled from the Twitter API using the tweet ids.

- [tweet_ids_with_hate_label.csv](Data/tweet_ids_with_hate_label.csv): contains retweet ids, root tweet ids, dummy usernames, and a hate speech label. The label is 1 if the root tweet is hate speech and 0 otherwise. Retweet ids are used in this work.
- [users_with_label.csv](Data/users_with_label.csv): contains dummy usernames and the probability to reshare hate speech as used in the causal framework.

## Code usage
### Input:
Run the script [create_bipartite.py](create_bipartite.py) to create the user-news bipartite graph. 
- Required data format for "user_attribute.pkl": {"anonymous_1": {"followers_count": int, "friend_count": int, "register_time": int, "status_count":int, "verified":int, "root_tweet_ids": \["root_tweet_id_1", "root_tweet_id_2", ...\], "label":float}, "anonymous_2": {...}, ...}
- Required data format for "root_tweets.csv"; pd.DataFrame with columns \["id", "label", "content"\]. "id" is the root tweet id (as int), "label" is the hate speech label (as int), and "content" is the root tweet content.

### IPS reweighting:
To get the virality- and followers-based propensity score estimation, run the script [pscore_V.py](pscore_V.py) and [pscore_F.py](pscore_F.py), respectively. 

### Modeling latent vulnerability:
For the recommendation models BPRMF, BPRMF_V and BPRMF_F, simply adjust the model type parameter in the config ([parser.py](Code/utility/parser.oy)) and run [BPRMF.py](Code/Model_BPRMF/BPRMF.py). For BPRMF_NN, adjust the config and run [BPRMF_neural.py](Code/Model_BPRMF/BPRMF_neural.py). 

### Estimating causal effects of user attributes:
To get the estimated causal effects using EBM:
1. Run [data_processing.py](Code/Causality/data_processing.py) to get the data in the required format.
2. Run [EBM.py](Code/Causality/EBM.py) to get the estimated causal effects.