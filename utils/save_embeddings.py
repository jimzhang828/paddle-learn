# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/03
"""
import json

import numpy as np
from tqdm import tqdm

# 获取词向量
VOCAB_SIZE = 0
EMBED_DIM = 0
EMBEDDING_PATH = './data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
OOV = '[OOV]'
vocab2id = {}
id2vocab = []
embeddings = None
with open(EMBEDDING_PATH) as f, tqdm(unit_scale=0, unit=' lines') as t:
    for i, line in enumerate(f):
        line = line.rstrip('\n').rstrip(' ')
        t.update(1)
        if i == 0:
            vocab2id[OOV] = i
            id2vocab.append(OOV)
            VOCAB_SIZE = int(line.split(' ')[0]) + 1  # +1: OOV
            EMBED_DIM = int(line.split(' ')[1])
            embeddings = np.zeros((VOCAB_SIZE, EMBED_DIM))
            continue
        entry = line.split(' ')
        assert len(entry) == EMBED_DIM + 1
        vocab2id[entry[0]] = i
        id2vocab.append(entry[0])
        embeddings[i, :] = [float(t) for t in entry[1:]]

np.save('./data/thucnews_embeddings.npy', embeddings)
with open('./data/thucnews_vocab2id.json', 'w+') as f1, open('./data/thucnews_id2vocab.json', 'w+') as f2:
    json.dump(vocab2id, f1)
    json.dump(id2vocab, f2)

