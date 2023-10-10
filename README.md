# Causal_hatespeech_dissemination

## Code usage
1. Run the script [preprocessing.py](preprocessing.py) to preprocess data.
2. Run the script [create_bipartite.py](create_bipartite.py) to creat the user-news bipartite graph.
3. To get the tweets- and user-tweets-based propensity score estimation, run the script [pscore.py](my_pscore.py) and [pscore_ut.py](my_pscore_ut.py), respectively. 
4. For the method BPRMF, BPRMF_H and BPRMF_UH, simply adjust the model type parameter in the config ([parser.py](Code/utility/parser.oy)) and run [BPRMF.py](Code/Model_BPRMF/BPRMF.py). For the BPRMF_Neural, adjust the config and run [BPRMF_neural.py](Code/Model_BPRMF/BPRMF_neural.py). They correspond to the biased model and unbiased models using tweets-, user-tweets-, and neural-network-based propensity score estimations. This also applies to the method NCF. 
5. Note that the main programs (BPRMF.py or NCF.py) mostly are adapted from code for paper [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108).
6. Causality analysis is conducted in [linear_regression.R](Code/Causality/regression/linear_regression.R).