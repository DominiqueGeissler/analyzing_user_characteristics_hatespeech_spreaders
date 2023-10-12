"""
Code adapted from "Causal Understanding of Fake News Dissemination on Social Media", by Cheng et. al in KDD 2021
Codes for preprocessing real-world datasets and computing propensity scores used in the experiments
"""
import codecs
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.model_selection import train_test_split
from Code.logger import get_logger


logger = get_logger(__name__)

logger.info("Compute virality-based propensity scores...")


"""Load and Preprocess datasets."""
logger.info("Load data...")
# load dataset.
col = {0: 'user', 1: 'item'}
with open(f'../../Data/train-pscore.txt', 'r') as f:
    data_train = pd.read_csv(f, delimiter=' ', header=None)
    data_train.rename(columns=col, inplace=True)
with open(f'../../Data/test-pscore.txt', 'r') as f:
    data_test = pd.read_csv(f, delimiter=' ', header=None)
    data_test.rename(columns=col, inplace=True)
num_users, num_items = data_train.user.max(), data_train.item.max()
logger.debug(f"Size of the training set: {len(data_train)}; size of the test set: {len(data_test)}")
logger.debug(f"Nr of users: {num_users}, nr of items: {num_items} in the training set.")

logger.info("Compute propensity scores...")
# train-test, split
train, test = data_train.values, data_test.values
combine=np.vstack((train,test))
combine=np.hstack((combine,np.ones((combine.shape[0],1))))
#train, val = train_test_split(train, test_size=0.1, random_state=12345)
# estimate pscore
_, item_freq = np.unique(combine[combine[:, 2] == 1, 1], return_counts=True)
pscore = (item_freq / item_freq.max()) ** 0.5

path = Path(f'../../Data')
path.mkdir(parents=True, exist_ok=True)
#np.save(str(path / 'train.npy'), arr=train.astype(np.int))
#np.save(str(path / 'val.npy'), arr=val.astype(np.int))
#np.save(str(path / 'test.npy'), arr=test.astype(np.int))
np.save(str(path / 'pscore_V.npy'), arr=pscore)
np.save(str(path / 'item_freq.npy'), arr=item_freq)
logger.info("Done.")
