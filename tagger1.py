import functools
import operator
from pprint import pprint
from dataloader import NER, POS
from torch.utils.data import DataLoader, IterableDataset
from sklearn.preprocessing import OneHotEncoder
from mlp import MLP
import numpy as np
import json
import pandas as pd

from word2vec import random_vocabulary, create_dataset

dataset = 'ner'
dataset_checkpoint = False

if dataset_checkpoint:
    with open(f'dataset/{dataset}/vocabulary.json', 'r') as file:
        vocabulary = json.load(file)

    with open(f'dataset/{dataset}/train.json', 'r') as file:
        train_dataset = json.load(file)
else:
    vocabulary = random_vocabulary(NER.train)
    # pd.DataFrame(vocabulary).to_json(f'dataset/{dataset}/vocabulary.json', orient='split')
    with open(f'dataset/{dataset}/vocabulary.json', 'w') as file:
        json.dump(vocabulary.tolist(), file)
    #
    # dataset = create_dataset(NER.train, vocabulary)
    #
    # with open(f'dataset/{dataset}/train.json', 'w') as file:
    #     json.dump(dataset, file)

# pprint(dataset)

# dataset = IterableDataset.from_generator(dataset_generator).with_format("torch")
#     data_loader = DataLoader(dataset, batch_size=8, num_workers=1, prefetch_factor=100, pin_memory=True)
# DataLoader(training_data, batch_size=64, shuffle=True)
#
# # words = {w[0].upper() for w in functools.reduce(operator.concat, POS.train)}
# # pos_vocabulary = {w: np.random.random(50) for w in words}