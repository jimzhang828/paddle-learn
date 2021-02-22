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


app = flask.Flask(__name__)
network = paddle.jit.load('./saved/jitmodel')
network.eval()
# x = paddle.to_tensor(np.array([0] * 32).reshape(1, 32))
# out = network(x)
# print(out.numpy().argmax())
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
    out = network(tokens)
    pred = out.numpy().argmax()
    return '{"pred": %d}' % pred


@app.route('/feed_class_text', methods=['POST'])
def feed_class_text():
    """
    feed class text
    """
    data = request.get_data()
    text = json.loads(data.decode('utf-8')).get('text', [])
    if not text:
        return '{"error": "no text input"}'
    tokens = jieba.cut(text)
    tokens = [vocab2id.get(t, 0) for t in tokens]
    tokens = np.array((tokens + [0] * 32)[:32]).reshape(1, 32)
    tokens = paddle.to_tensor(tokens)
    out = network(tokens)
    pred = out.numpy().argmax()
    return '{"pred": %d}' % pred


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9299, threaded=True)

