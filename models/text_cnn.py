# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/02
"""
import numpy as np
import paddle
import paddle.nn as nn
from paddle.fluid.initializer import NumpyArrayInitializer


class TextCNN(nn.Layer):
    """
    TextCNN
    """
    def __init__(self, vocab_size, embed_dim, num_class, kernel_num, kernel_sizes, dropout, embeddings=None):
        super(TextCNN, self).__init__()
        
        weight_attr = None
        if embeddings is not None:
            weight_attr = paddle.ParamAttr(name='embeddings', 
                                           initializer=NumpyArrayInitializer(embeddings), 
                                           learning_rate=0.5, 
                                           trainable=True)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True, weight_attr=weight_attr)
        layer_list = []
        for k_size in kernel_sizes:
            layer_list.append(nn.Conv2D(in_channels=1, 
                                        out_channels=kernel_num, 
                                        kernel_size=(k_size, embed_dim)))
        self.conv_list = nn.LayerList(layer_list)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, num_class)

    def forward(self, x):
        """
        forward
        """
        x = self.embedding(x)  # (16, 128) -> (16, 128, 100)
        # x = paddle.unsqueeze(x, axis=1)
        x = x.unsqueeze(axis=1)  # (16, 1, 128, 100)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.conv_list]
        x = [nn.functional.avg_pool1d(i, i.shape[2]).squeeze(2) for i in x]
        x = paddle.concat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = TextCNN(vocab_size=10000, embed_dim=100, num_class=10, 
                  kernel_num=100, kernel_sizes=[2, 3, 4], dropout=0.5)
    paddle.summary(net, (16, 128), dtypes=np.int32)

