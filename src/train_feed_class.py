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
import paddle_serving_client.io as serving_io

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
with open(VOCAB_PATH) as f:
    vocab2id = json.load(f)
id2vocab = {v: k for k, v in vocab2id.items()}
NUM_CLASS = len(id2label)
embeddings = np.load(EMBEDDING_PATH)
VOCAB_SIZE, EMBED_DIM = embeddings.shape
# VOCAB_SIZE = len(vocab2id)
# EMBED_DIM = 300
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


# def _setup_datasets(train_path, valid_path, test_path):
#     """处理数据集"""
#     # 创建训练集
#     logging.info('Creating training data')
#     train_data = _create_dataset(train_path)

#     # 创建验证集
#     logging.info('Creating validation data')
#     valid_data = _create_dataset(valid_path)

#     # 创建测试集
#     logging.info('Creating testing data')
#     test_data = _create_dataset(test_path)
#     # test_data = None

#     return train_data, valid_data, test_data


def token_padding(text, length):
    """对输入text进行padding，长则截取，短则补0"""
    return (text + [0] * length)[0: length]


def generate_batch(batch):
    """collate_fn"""
    text = paddle.to_tensor([token_padding(entry[0], MAX_SEQUENCE_LENGTH) for entry in batch])
    label = paddle.to_tensor([entry[1] for entry in batch])
    return text, label


# train_data, valid_data, test_data = _setup_datasets(TRAIN_PATH, VALID_PATH, TEST_PATH)
train_data = _create_dataset(TRAIN_PATH)
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=generate_batch)
valid_data = _create_dataset(VALID_PATH)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)
# test_data = _create_dataset(TEST_PATH)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)

model_input = InputSpec([None, MAX_SEQUENCE_LENGTH], 'int64', 'input')
model_label = InputSpec([None, 1], 'int64', 'label')

network = TextCNN(vocab_size=VOCAB_SIZE, 
                  embed_dim=EMBED_DIM, 
                  num_class=NUM_CLASS, 
                  kernel_num=KERNEL_NUM, 
                  kernel_sizes=KERNEL_SIZES, 
                  dropout=DROPOUT, 
                  embeddings=embeddings)

# Model 方式训练
# model = paddle.Model(network, inputs=model_input, labels=model_label)
# loss_fn = paddle.nn.CrossEntropyLoss()
# optimizer = paddle.optimizer.SGD(learning_rate=LR, parameters=model.parameters())
# metrics = [
#     paddle.metric.Accuracy()
# ]
# model.prepare(optimizer=optimizer, loss=loss_fn, metrics=metrics)
# model.fit(train_data=train_loader, eval_data=valid_loader, epochs=NUM_EPOCHS, verbose=1)

# model.save('saved/checkpoint')
# model.save('saved/inference_model')

# logging.info('loading model')
# model.load('saved/checkpoint')
# logging.info('load model successfully')

# model.evaluate(train_loader)
# model.evaluate(valid_loader)
# model.evaluate(test_loader)

# logging.info('start saving model')
# paddle.jit.save(model.network, './saved/jitmodel')
# serving_io.save_dygraph_model('./saved/serving_model', './saved/client_config', model.network)
# logging.info('saving model successfully')

# 基础方式训练
network.train()
loss_fn = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.SGD(learning_rate=LR, parameters=network.parameters())
for epoch in range(NUM_EPOCHS):
    for batch_id, (x_data, y_data) in enumerate(train_loader()):

        y_data = paddle.reshape(y_data, [-1, 1])

        predicts = network(x_data)    # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        # acc = paddle.metric.accuracy(predicts, y_data)

        # 反向传播
        loss.backward()

        if (batch_id + 1) % 100 == 0:
            logging.info("epoch=%d, batch=%d, loss=%f", epoch, batch_id + 1, loss.numpy())

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.clear_grad()
    
    # Evaluation
    network.eval()
    all_predicts, all_labels = None, None
    for batch_id, (x_data, y_data) in enumerate(valid_loader):
        y_data = paddle.reshape(y_data, [-1, 1])
        predicts = network(x_data)
        if all_predicts is None:
            all_predicts = predicts
            all_labels = y_data
        else:
            all_predicts = paddle.concat([all_predicts, predicts], axis=0)
            all_labels = paddle.concat([all_labels, y_data], axis=0)
    loss = loss_fn(all_predicts, all_labels)
    acc = paddle.metric.accuracy(all_predicts, all_labels)
    logging.info("complete epoch=%d, loss=%f, acc=%f", epoch, loss.numpy(), acc.numpy())
    network.train()


logging.info('start saving model')
paddle.jit.save(network, './saved/feed_class_0223/jitmodel')
serving_io.save_dygraph_model('./saved/feed_class_0223/serving_model', './saved/feed_class_0223/client_config', network)
logging.info('saving model successfully')


def main():
    """
    main
    """


if __name__ == '__main__':
    main()
