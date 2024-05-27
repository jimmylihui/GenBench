import torch
import torch.nn as nn

from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

def ResidualBlock(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))

class SpliceAI(nn.Module):
    S = 2000

    def __init__(self,input_length,output_length):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
        )

        self.res_conv3 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            nn.Conv1d(32, 32, 1, dilation=1, padding='same')
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding='same')

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x)
        detour += self.res_conv3(x)

        x = self.block3(x) + detour
        x = self.conv_last(x)

        return rearrange(x[..., :self.output_length], 'b c l -> b l c')