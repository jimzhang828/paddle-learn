# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/21
"""
import json

import flask
from flask import request
import paddle
import numpy as np
import jieba
import paddle.static as static
import threading

lock = threading.Lock()
exec_strategy = static.ExecutionStrategy()
exec_strategy.num_threads = 4

app = flask.Flask(__name__)
network = paddle.jit.load('./saved/jitmodel')
network.eval()

with open('./data/thucnews_vocab2id.json') as f:
    vocab2id = json.load(f)


@app.route('/feed_class_tokens', methods=['POST'])
def feed_class_tokens():
    """
    feed class tokens
    """
    data = request.get_data()
    tokens = json.loads(data.decode('utf-8')).get('tokens', [])
    if not tokens:
        return '{"error": "no tokens input"}'
    tokens = paddle.to_tensor(np.array(tokens).reshape(1, 32))
    with lock:
        out = network(tokens)
    pred = out.numpy().argmax()
    return '{"pred": %d}' % pred


@app.route('/feed_class_batch_tokens', methods=['POST'])
def feed_class_batch_tokens():
    """
    feed class batch tokens
    """
    data = request.get_data()
    batch_tokens = json.loads(data.decode('utf-8')).get('batch_tokens', [])
    if not batch_tokens:
        return '{"error": "no batch_tokens input"}'
    batch_tokens = paddle.to_tensor(np.array(batch_tokens))
    with lock:
        out = network(batch_tokens)
    pred = out.numpy().argmax(axis=1)
    return '{"pred": [%s]}' % ','.join(map(str, pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9299, threaded=True)

