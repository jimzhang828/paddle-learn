# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/21
"""
import json

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
        self.client = Client()
        self.connect_to_servers()
    
    def connect_to_servers(self):
        """
        连接到server
        """
        self.client.load_client_config(self.config_file)
        self.client.connect(self.urls)

    def predict(self, tokens):
        """
        预测
        """
        fetch_map = self.client.predict(feed={self.feed_var: tokens}, fetch=[self.fetch_var])
        logits = fetch_map[self.fetch_var]
        return np.argmax(logits)

    def predict_batch(self, batch_tokens):
        """
        batch预测
        """
        fetch_map = self.client.predict(feed={self.feed_var: batch_tokens}, fetch=[self.fetch_var], batch=True)
        logits = fetch_map[self.fetch_var]
        return np.argmax(logits, axis=1)


if __name__ == '__main__':
    client = FeedClassificationClient('saved/client_config/serving_client_conf.prototxt', ['127.0.0.1:9393'])
    # client = FeedClassificationClient(
    #     'saved/client_config/serving_client_conf.prototxt', 
    #     ['yq01-bdg-dst-gpu-k40m-03.yq01.baidu.com:9393'])
    batch_tokens = np.array([[0] * 32] * 16)
    tokens = np.array([0] * 32)
    print(client.predict(tokens))
    print(client.predict_batch(batch_tokens))
