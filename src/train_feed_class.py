# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/03
"""
import os
import json
import logging
import configparser

import numpy as np
import jieba
import paddle
from paddle.static import InputSpec
from paddle.io import DataLoader
from tqdm import tqdm

import utils.log as log
from models.text_cnn import TextCNN
from datasets.feed_class_dataset import FeedClassDataset
log.init_log('log/train_feed_class')


# 读取配置
config = configparser.ConfigParser()
config.read('config/textcnn_feed_class.conf')

TRAIN_PATH = config.get('data', 'train_path')
VALID_PATH = config.get('data', 'valid_path')
TEST_PATH = config.get('data', 'test_path')
EMBEDDING_PATH = config.get('data', 'embedding_path')
VOCAB_PATH = config.get('data', 'vocab_path')
LABELS = config.get('data', 'labels')

MAX_SEQUENCE_LENGTH = int(config.get('train', 'max_sequence_length'))
BATCH_SIZE = int(config.get('train', 'batch_size'))
NUM_EPOCHS = int(config.get('train', 'num_epochs'))
LR = float(config.get('train', 'learning_rate'))

KERNEL_NUM = int(config.get('model', 'kernel_num'))
KERNEL_SIZES = [int(k_size) for k_size in config.get('model', 'kernel_sizes').split(',')]
DROPOUT = float(config.get('model', 'dropout_rate'))

STEP_SIZE = int(config.get('scheduler', 'step_size'))
GAMMA = float(config.get('scheduler', 'gamma'))

id2label = LABELS.split(',')
label2id = {label: i for i, label in enumerate(id2label)}
embeddings = np.load(EMBEDDING_PATH)
with open(VOCAB_PATH) as f:
    vocab2id = json.load(f)
id2vocab = {v: k for k, v in vocab2id.items()}
NUM_CLASS = len(id2label)
VOCAB_SIZE, EMBED_DIM = embeddings.shape
logging.info('embedding shape: (%d, %d)', VOCAB_SIZE, EMBED_DIM)
logging.info('labels: %s', ','.join(['%d:%s' % (i, label) for i, label in enumerate(id2label)]))


def _create_dataset(data_path):
    """
    传入数据集文件路径，返回Paddle数据集
    数据集每行格式为 label\tcontent
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
            label = label2id[entry[0]]
            data.append((tokens, label))
    dataset = FeedClassDataset(data=data)
    return dataset


def _setup_datasets(train_path, valid_path, test_path):
    """处理数据集"""
    # 创建训练集
    logging.info('Creating training data')
    train_data = _create_dataset(train_path)

    # 创建验证集
    logging.info('Creating validation data')
    valid_data = _create_dataset(valid_path)

    # 创建测试集
    logging.info('Creating testing data')
    # test_data = _create_dataset(test_path)
    test_data = None

    return train_data, valid_data, test_data


def token_padding(text, length):
    """对输入text进行padding，长则截取，短则补0"""
    return (text + [0] * length)[0: length]


def generate_batch(batch):
    """collate_fn"""
    text = paddle.to_tensor([token_padding(entry[0], MAX_SEQUENCE_LENGTH) for entry in batch])
    label = paddle.to_tensor([entry[1] for entry in batch])
    return text, label


train_data, valid_data, test_data = _setup_datasets(TRAIN_PATH, VALID_PATH, TEST_PATH)
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=generate_batch)
valid_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)

model_input = InputSpec([None, MAX_SEQUENCE_LENGTH], 'int64', 'input')
model_label = InputSpec([None, 1], 'int64', 'label')

network = TextCNN(vocab_size=VOCAB_SIZE, 
                  embed_dim=EMBED_DIM, 
                  num_class=NUM_CLASS, 
                  kernel_num=KERNEL_NUM, 
                  kernel_sizes=KERNEL_SIZES, 
                  dropout=DROPOUT, 
                  embeddings=embeddings)
model = paddle.Model(network, inputs=model_input, labels=model_label)
loss_fn = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.SGD(learning_rate=LR, parameters=model.parameters())
metrics = [
    # paddle.metric.Precision(), 
    # paddle.metric.Recall(), 
    paddle.metric.Accuracy()
]
model.prepare(optimizer=optimizer, loss=loss_fn, metrics=metrics)
model.fit(train_data=train_loader, eval_data=valid_loader, epochs=NUM_EPOCHS, verbose=1, save_dir='./saved')


def main():
    """
    main
    """


if __name__ == '__main__':
    main()
