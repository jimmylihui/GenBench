
import torch.nn as nn
import torch

class DeepSTAR(nn.Module):
    def __init__(self,input_size,output_size):
        super(DeepSTAR,self).__init__()
        self.embedding=nn.Sequential(
            nn.Conv1d(4,256,kernel_size=7,padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.hidden_layer_1=nn.Sequential(
            nn.Conv1d(256,60,kernel_size=3,padding='same'),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.hidden_layer_2=nn.Sequential(
            nn.Conv1d(60,60,kernel_size=5,padding='same'),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.hidden_layer_3=nn.Sequential(
            nn.Conv1d(60,60,kernel_size=5,padding='same'),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.hidden_layer_4=nn.Sequential(
            nn.Conv1d(60,120,kernel_size=3,padding='same'),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.dense_layers_1=nn.Sequential(
            nn.Conv1d(120,256,kernel_size=1),
            
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.dense_layers_2=nn.Sequential(
            nn.Conv1d(256,256,kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.output_layer=nn.Linear(256,output_size)

    def forward(self,x):
        x=x.permute(0,2,1)
        
        x=self.embedding(x)
        x=self.hidden_layer_1(x)
        x=self.hidden_layer_2(x)
        x=self.hidden_layer_3(x)
        x=self.hidden_layer_4(x)

        x=self.dense_layers_1(x)
        x=self.dense_layers_2(x)
        x=x.permute(0,2,1)

        x=x.mean(dim=1,keepdim=False)
        x=self.output_layer(x)
        
        return x
        