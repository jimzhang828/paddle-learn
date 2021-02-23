# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/22
"""
import json
import time
import threading

import jieba
import requests
import numpy as np
from tqdm import tqdm
from sklearn import metrics

with open('./data/thucnews_vocab2id.json') as f:
    vocab2id = json.load(f)
id2label = '军事,社会,娱乐,游戏,体育,教育,旅游,财经,房产,国际,科学,汽车,历史,悦读'.split(',')
label2id = {label: i for i, label in enumerate(id2label)}


def _create_data(data_path):
    """
    传入数据集文件路径，返回TOKENS-LABEL对
    """
    data = []
    with open(data_path) as f, tqdm(unit_scale=0, unit=' lines') as t:
        for line in f:
            t.update(1)
            entry = line.split('\t')
            if len(entry) < 2:
                continue
            tokens = list(jieba.cut(entry[1]))
            tokens = [vocab2id.get(t, 0) for t in tokens]
            tokens = (tokens + [0] * 32)[:32]
            label = label2id[entry[0]]
            data.append((tokens, label))
            if len(data) >= 8000:
                break
    return data


class TestThread(threading.Thread):
    """test thread"""
    def __init__(self, data):
        super(TestThread, self).__init__()
        self.test_data = data
    
    def run(self):
        """run"""
        y_true = []
        y_pred = []
        t_start = time.time() * 1000
        batch_tokens = []
        batch_label = []
        for tokens, label in self.test_data:
            batch_tokens.append(tokens)
            batch_label.append(label)
            if len(batch_label) < 1400:
                continue
            data = json.dumps({'batch_tokens': batch_tokens})
            resp = requests.post('http://localhost:9299/feed_class_batch_tokens', 
                data=data.encode('utf-8'), 
                headers={'Content-Type': 'application/json'})
            batch_pred = json.loads(resp.text)['pred']
            for p, t in zip(batch_pred, batch_label):
                y_true.append(t)
                y_pred.append(p)
            batch_tokens.clear()
            batch_label.clear()
        t_end = time.time() * 1000
        print('elapse: %d ms' % (t_end - t_start))
        print('acc: %f' % metrics.accuracy_score(y_true, y_pred))
        print('pre: %f' % metrics.precision_score(y_true, y_pred, average='macro'))
        print('rec: %f' % metrics.recall_score(y_true, y_pred, average='macro'))
        print('f1: %f' % metrics.f1_score(y_true, y_pred, average='macro'))


if __name__ == '__main__':
    test_data = _create_data('./data/text_1000/test.txt')
    threads = []
    for _ in range(8):
        t = TestThread(test_data)
        threads.append(t)
    # run
    for t in threads:
        t.start()
    for t in threads:
        t.join()
