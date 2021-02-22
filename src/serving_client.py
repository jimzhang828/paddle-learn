# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/21
"""
import json

import jieba
import numpy as np
from paddle_serving_client import Client


class FeedClassificationClient(object):
    """
    Feed classification client
    """
    def __init__(self, config_file: str, urls: list):
        self.config_file = config_file
        self.urls = urls
        self.feed_var = 'generated_var_17144'
        self.fetch_var = 'translated_layer/scale_0.tmp_0'
        with open('./data/thucnews_vocab2id.json') as f:
            self.vocab2id = json.load(f)
        self.client = Client()
        self.connect_to_servers()
    
    def connect_to_servers(self):
        """
        连接到server
        """
        self.client.load_client_config(self.config_file)
        self.client.connect(self.urls)
    
    def tokenize(self, text):
        """tokenization"""
        tokens = jieba.cut(text)
        tokens = [self.vocab2id.get(t, 0) for t in tokens]
        tokens = np.array((tokens + [0] * 32)[:32])
        return tokens
    
    def predict(self, text):
        """
        预测
        """
        tokens = self.tokenize(text)
        fetch_map = self.client.predict(feed={self.feed_var: tokens}, fetch=[self.fetch_var])
        logits = fetch_map[self.fetch_var]
        return np.argmax(logits)


if __name__ == '__main__':
    client = FeedClassificationClient('saved/client_config/serving_client_conf.prototxt', ['127.0.0.1:9292'])
    text = '百度你好'
    print(client.predict(text))
