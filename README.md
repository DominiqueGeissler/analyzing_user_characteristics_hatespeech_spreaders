# Analyzing User Characteristics of Hate Speech Spreaders on Social Media

This is the code for the paper A Debiasing Framework for Analyzing Hate Speech Resharing.

The code has been tested on Python 3.8

## Abstract
Hate speech on social media threatens the mental and physical well-being of individuals and contributes to real-world violence. Resharing is an important driver behind the spread of hate speech on social media. Yet, little is known about who reshares hate speech and what their characteristics are. In this paper, we analyze the role of user characteristics in hate speech resharing across different types of hate speech (e.g., political hate). For this, we proceed as follows: First, we cluster hate speech posts using large language models to identify different types of hate speech. Then we model the effects of user attributes on users' probability to reshare hate speech using an explainable machine learning model. To do so, we apply debiasing to control for selection bias in our observational social media data and further control for the latent vulnerability of users to hate speech. We find that, all else equal, users with fewer followers, fewer friends, fewer posts, and older accounts share more hate speech. This shows that users with little social influence tend to share more hate speech. Further, we find substantial heterogeneity across different types of hate speech. For example, racist and misogynistic hate is spread mostly by users with little social influence. In contrast, political anti-Trump and anti-right-wing hate is reshared by users with larger social influence. Overall, understanding the factors that drive users to share hate speech is crucial for detecting individuals at risk of engaging in harmful behavior and for designing effective mitigation strategies.
## Repository Structure
Code: Folder to store the code used for the analysis.\
Data: Folder to store the data.
- source: store user attributes and root tweets there to get started.

## Data
We anonymize the usernames by replacing them with dummy usernames. User attributes can be crawled from the Twitter API using the tweet ids.

- [tweet_ids_with_hate_label.csv](Data/tweet_ids_with_hate_label.csv): contains retweet ids, root tweet ids, dummy usernames, and a hate speech label. The label is 1 if the root tweet is hate speech and 0 otherwise. Retweet ids are used in this work.
- [users_with_label.csv](Data/users_with_label.csv): contains dummy usernames and the probability to reshare hate speech as used in the debiasing framework.

## Code usage
### Input:
Run the script [create_bipartite.py](create_bipartite.py) to create the user-news bipartite graph. 
- Required data format for "user_attribute.pkl": {"anonymous_1": {"followers_count": int, "friend_count": int, "register_time": int, "status_count":int, "verified":int, "root_tweet_ids": \["root_tweet_id_1", "root_tweet_id_2", ...\], "label":float}, "anonymous_2": {...}, ...}
- Required data format for "root_tweets.csv"; pd.DataFrame with columns \["id", "label", "content"\]. "id" is the root tweet id (as int), "label" is the hate speech label (as int), and "content" is the root tweet content.

### Clustering hate speech:
To cluster the hate speech posts, run the script [topic_grouping.py](Code/Clustering/topic_grouping.py). 
You can name the hate speech clusters by running [topic_labeling.py](Code/Clustering/topic_labeling.py).

### IPS reweighting:
To get the virality- and followers-based propensity score estimation, run the script [pscore_V.py](pscore_V.py) and [pscore_F.py](pscore_F.py), respectively. 

### Modeling past latent vulnerability:
For the recommendation models BPRMF, BPRMF_V and BPRMF_F, simply adjust the model type parameter in the config ([parser.py](Code/utility/parser.oy)) and run [BPRMF.py](Code/Model_BPRMF/BPRMF.py). For BPRMF_NN, adjust the config and run [BPRMF_neural.py](Code/Model_BPRMF/BPRMF_neural.py). 

### Estimating effects of user attributes:
To get the estimated effects using EBM:
1. Run [data_processing.py](Code/Causality/data_processing.py) to get the data in the required format.
2. Run [EBM.py](Code/Causality/EBM.py) to get the estimated effects.

### Establishing causality:
1. Run [semi-synthetic_evaluation.ipynb](Code/Causality/semi-synthetic_evaluation.ipynb) to get the semi-synthetic data and evaluate performance on counterfactuals.
2. Run [sensitivity_analysis.ipynb](Code/Causality/sensitivity_analysis.ipynb) to get the sensitivity analysis results.