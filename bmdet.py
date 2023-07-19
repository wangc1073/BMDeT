import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import BayesianLinear
from utils import variational_estimator
import copy
import math

################################################################################################################

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
###########################

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h

        self.h = h
        self.linears = clones(BayesianLinear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value,dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
####################

def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
###############################

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#####################

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
############################

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = BayesianLinear(d_model, d_ff)
        self.w_2 = BayesianLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
#####################

class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(EncoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=128, dropout=dropout)
        self.sublayer = clones(SublayerConnection(size=d_model, dropout=dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
###############################

class Encoder(nn.Module):
    def __init__(self, N, d_model):
        super(Encoder, self).__init__()
        layer = EncoderLayer(d_model=d_model,dropout=0.1)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
###############################

class DecoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=dropout)
        self.src_attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=128, dropout=dropout)
        self.sublayer = clones(SublayerConnection(size=d_model, dropout=dropout), 3)

    def forward(self, memory, x):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)
#########################

class Decoder(nn.Module):
    def __init__(self, N, d_model,ahead):
        super(Decoder, self).__init__()
        layer = DecoderLayer(d_model=d_model,dropout=0.1)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.outputLinear = BayesianLinear(d_model, 16)
        self.outputLinear1 = BayesianLinear(72*16, 72)
        self.outputLinear2 = BayesianLinear(72, ahead)

    def forward(self, memory, decoderINPUT):
        for layer in self.layers:
            memory = layer(memory, decoderINPUT)
        return self.outputLinear2(self.outputLinear1(torch.flatten(input = self.outputLinear(self.norm(memory)),start_dim=1)))
#################

@variational_estimator
class BayesianMDeT(nn.Module):
    """
    BayesianDeT
    """
    def __init__(self,ahead):
        super(BayesianMDeT, self).__init__()
        d_model = 32
        self.encoder = Encoder(2,d_model=d_model)
        self.decoder1 = Decoder(2,d_model=d_model,ahead=ahead)
        self.decoder2 = Decoder(2,d_model=d_model,ahead=ahead)
        self.decoder3 = Decoder(2,d_model=d_model,ahead=ahead)

        self.encoderLinear = BayesianLinear(16, d_model)
        self.decoder1_Linear = BayesianLinear(14, d_model)
        self.decoder2_Linear = BayesianLinear(14, d_model)
        self.decoder3_Linear = BayesianLinear(14, d_model)

    def forward(self, x):
        # x: input data. shape (batch, 72, 16)
        encoder_output = self.encoder(self.encoderLinear(x))
        # aux: Auxiliary information. shape (batch, 72, 13)
        aux = x[ : , : , 3 : ]
        # inputi: load data + Auxiliary information. shape (batch, 72, 14)
        input1 = torch.cat([x[ : , : ,   : 1 ],aux], dim = 2)
        input2 = torch.cat([x[ : , : , 1 : 2 ],aux], dim = 2)
        input3 = torch.cat([x[ : , : , 2 : 3 ],aux], dim = 2)

        decoder1_output = self.decoder1(encoder_output, self.decoder1_Linear(input1))
        decoder2_output = self.decoder2(encoder_output, self.decoder2_Linear(input2))
        decoder3_output = self.decoder3(encoder_output, self.decoder3_Linear(input3))

        return decoder1_output, decoder2_output, decoder3_output
#################
