

import os

import numpy as np

import torch
from torch import nn
from sklearn import metrics


# A simple CNN model
class CNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, embedding_dim, input_len):
        super(CNN, self).__init__()


        
        number_of_output_neurons = number_of_classes
        minimum_length = 70
        output_activation = lambda x: x
        
        
        self.padding_flag = True
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if input_len < minimum_length:
            self.padding = nn.ConstantPad1d((0, minimum_length - input_len), 0)
            input_len = minimum_length
        self.cnn_model = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=8, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=8, bias=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=8, bias=True),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(2),

            nn.Flatten()
        )
        self.dense_model = nn.Sequential(
            nn.Linear(self.count_flatten_size(input_len), 512),
            nn.Linear(512, number_of_output_neurons)
        )
        self.output_activation = output_activation


    def count_flatten_size(self, input_len):
        zeros = torch.zeros([1, input_len], dtype=torch.long)
        x = self.embeddings(zeros)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        return x.size()[1]

    def forward(self, x):
        if self.padding_flag:
            x = self.padding(x)
        x = self.embeddings(x)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        x = self.dense_model(x)
        x = self.output_activation(x)
        return x

if __name__ == '__main__':
    model = CNN(2, 1000, 256, 30)
    x = torch.randint(0, 1000, [1, 30])
    y = model(x)
    print(y.shape)
