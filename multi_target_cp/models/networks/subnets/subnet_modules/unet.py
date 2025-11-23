#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class UNet(nn.Module):
    
    def __init__(self, c_in, c_out, num_filters = [32,64]):
        super().__init__()
        
        self.num_filters = num_filters
        
        #Encoder convolution blocks
        self.encode_conv_blocks = [Conv_Block(c_in, num_filters[0])]
        for i in range(1,len(num_filters)):
            self.encode_conv_blocks.append(Conv_Block(num_filters[i-1], num_filters[i]))
        self.encode_conv_blocks = nn.ModuleList(self.encode_conv_blocks)
            
        #Decoder convolution blocks
        #self.decode_conv_blocks = [Conv_Block(2*num_filters[i], num_filters[i-1]) for i in reversed(range(1, len(num_filters)))]
        self.decode_conv_blocks = []
        self.decode_conv_blocks.append(Conv_Block(2*num_filters[-1], num_filters[-1]))
        for i in reversed(range(0, len(num_filters)-1)):
            self.decode_conv_blocks.append(Conv_Block(num_filters[i+1] + num_filters[i], num_filters[i]))
        self.decode_conv_blocks = nn.ModuleList(self.decode_conv_blocks)
        
        
        self.conv_block_bridge = Conv_Block(self.num_filters[-1], self.num_filters[-1])
        self.maxpool = nn.MaxPool2d((2,2))
        self.upsample = nn.Upsample(scale_factor = (2,2))
        self.last_conv = nn.Conv2d(num_filters[0], c_out, kernel_size = (1,1), bias=False)
        self.act = nn.Sigmoid()
        
        self.last_conv.weight.data.zero_()
        
        
        
    
    def forward(self, x):
        
        skip_x = []
        #track = [x]
        #print(x.size())
        #Encoding
        for i in range(len(self.num_filters)):
            x = self.encode_conv_blocks[i](x)
            #track.append(x)
            skip_x.append(x)
            x = self.maxpool(x)
            #track.append(x)
            #print(x.size())
         
        
        
        #Bridge
        x = self.conv_block_bridge(x)
        
        #Reverse direction of the skip connections
        skip_x = skip_x[::-1]
        
        #Decoding
        for i in range(len(self.num_filters)):
            x = self.upsample(x)
            #print(x.size())
            #print(skip_x[i].size())
            x = torch.concat([x, skip_x[i]], dim=1)
            x = self.decode_conv_blocks[i](x)
            #print(x.size())
            
        #Output 
        #Get the channels to be what we want
        x = self.last_conv(x)
        x = self.act(x)
        #print(x.size())
        
        return x
    
    

class Conv_Block(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        
        self.conv_seq = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            )
        
        self.conv_seq.apply(init_weights)
        
        
    def forward(self,x):
        return self.conv_seq(x)
    
    
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        #m.bias.data.fill_(0.01)    