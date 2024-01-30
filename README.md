# A Debiasing Framework for Analyzing Hate Speech Resharing

This is the code for the paper A Debiasing Framework for Analyzing Hate Speech Resharing.

The code has been tested on Python 3.8

## Abstract
Hate speech on social media threatens the mental and physical well-being of individuals and is further responsible for real-world violence. An important driver behind the spread of hate speech and thus why hateful posts can go viral are reshares, yet little is known about who reshares hate speech and what their characteristics are. However, analyzing user attributes that make users share hate speech is challenging for two reasons: observational social media data are likely to suffer from selection bias, and the vulnerability of users to hate speech may vary, which leads to biased estimates. We thus develop a novel, debiasing framework with three steps: (1) We debias our observational social media data by applying inverse propensity scoring to account for selection bias. (2) We then use the debiased propensity scores to model the past latent vulnerability of users to hate speech as a latent embedding, which we control for in our analysis. (3) We model the effects of user attributes on users' probability of sharing hate speech using an explainable machine learning model. Compared to existing baselines, a particular strength of our framework is that it models relationships that are non-linear, yet still explainable. We find that users with fewer followers, fewer friends, and fewer posts share more hate speech. Younger accounts, in return, share less hate speech. Overall, understanding the factors that drive users to share hate speech is crucial for detecting individuals at risk of engaging in harmful behavior and for designing effective mitigation strategies.

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

### Step 1: IPS reweighting:
To get the virality- and followers-based propensity score estimation, run the script [pscore_V.py](pscore_V.py) and [pscore_F.py](pscore_F.py), respectively. 

### Step 2: Modeling past latent vulnerability:
For the recommendation models BPRMF, BPRMF_V and BPRMF_F, simply adjust the model type parameter in the config ([parser.py](Code/utility/parser.oy)) and run [BPRMF.py](Code/Model_BPRMF/BPRMF.py). For BPRMF_NN, adjust the config and run [BPRMF_neural.py](Code/Model_BPRMF/BPRMF_neural.py). 

### Step 3: Estimating effects of user attributes:
To get the estimated effects using EBM:
1. Run [data_processing.py](Code/Causality/data_processing.py) to get the data in the required format.
2. Run [EBM.py](Code/Causality/EBM.py) to get the estimated effects.

### Establishing causality:
1. Run [semi-synthetic_evaluation.ipynb](Code/Causality/semi-synthetic_evaluation.ipynb) to get the semi-synthetic data and evaluate performance on counterfactuals.
2. Run [sensitivity_analysis.ipynb](Code/Causality/sensitivity_analysis.ipynb) to get the sensitivity analysis results.