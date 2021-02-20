# coding=utf-8
"""
Author:  ZHANG Wenjun
Date:    2021/02/03
"""
from paddle.io import Dataset


class FeedClassDataset(Dataset):
    """
    feed标题分类数据集
    """
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
