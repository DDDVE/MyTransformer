"""define transformer"""

from math import sqrt
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

myInf = 10e12

# 返回position embedding中的w值
def position_w(k:int, dim:int):
    return 1.0 / (10000 ** ((2 * k) / dim))

class MyTransformerConfig:
    '''A: dim_qkv * n_head = embed_size'''
    max_sequence_len:   int = 1024              # 模型最大序列长度
    vocab_size:         int = 55500
    embed_size:         int = 256               # 每个token编码后的维度
    fix_pos_embed:      bool= True              # 是否使用fix position embedding 
    n_head:             int = 8                 # 多头注意力
    dim_qkv:            int = 0                 # 生成的QKV矩阵最后一维
    N:                  int = 3                 # block数量

'''
embedding
'''
# 包含input embedding和position embedding
# 输出两种embedding的和
class MyEmbedding(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.input_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        '''Q: figure out embedding'''
        '''A: give each possible word a vector with dimension embed_size which is much smaller than vocab_size'''
        '''A: comparing to one-oht encoding, it is smaller and trainable'''
        self.learn_position_embedding = nn.Embedding(config.max_sequence_len, config.embed_size)
        self.config = config
        self.register_buffer("fix_embedding", self.fix_position_embedding())
        
        '''init param'''

    # 固定参数的正余弦position embedding
    # return: max_sequence_len * embed_size
    def fix_position_embedding(self):
        output = torch.zeros((self.config.max_sequence_len, self.config.embed_size), dtype=torch.float)
        for j in range(self.config.max_sequence_len):
            for k in range(self.config.embed_size):
                if (k % 2 == 0):
                    output[j][k] = math.sin(j * position_w(k/2, self.config.embed_size))
                else:
                    output[j][k] = math.cos(j * position_w((k-1)/2, self.config.embed_size))
        return output

    # x: batch_size * max_sequence_len
    # return: batch_size * max_sequence_len * embed_size
    def forward(self, x):
        a = self.input_embedding(x)
        '''Q: why author finally chose fix embedding ?'''
        '''A: this kind MAY infer with longer sequence'''
        if self.config.fix_pos_embed:
            b = [self.fix_embedding] * x.size(dim=0)
            b = torch.stack(b, dim=0)
        else:
            b = self.learn_position_embedding(x)
        return a + b

'''
encoder self attention

padding mask
'''
class MyMultiAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.multi_q = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.multi_k = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.multi_v = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.linear = nn.Linear(config.dim_qkv * config.n_head, config.embed_size)
        self.dim_model = config.dim_qkv
        self.config = config
        
        '''init param'''
        torch.nn.init.normal_(self.multi_q.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.multi_k.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.multi_v.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    # x: batch_size * max_sequence_len * embed_size
    # padding_pos: batch_size * max_sequence_len
    def forward(self, x, padding_pos):
        batch_size = x.size(dim=0)
        multi_q = self.multi_q(x)   # batch_size * max_sequence_len * (dim_qkv * n_head)
        multi_k = self.multi_k(x)
        multi_v = self.multi_v(x)

        '''do reshape and transpose'''
        multi_q = multi_q.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.config.dim_qkv)
        multi_q = multi_q.transpose(1, 2)   # batch_size * n_head * max_sequence_len * dim_qkv
        multi_k = multi_k.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.config.dim_qkv)
        multi_k = multi_k.transpose(1, 2)   # batch_size * n_head * max_sequence_len * dim_qkv
        multi_v = multi_v.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.config.dim_qkv)
        multi_v = multi_v.transpose(1, 2)   # batch_size * n_head * max_sequence_len * dim_qkv

        att = multi_q @ (multi_k.transpose(-1, -2))
        att = att / math.sqrt(self.dim_model)   # batch_size * n_head * max_sequence_len * max_sequence_len
        '''padding mask'''
        '''same batch use the same padding pos'''
        for i in range(batch_size):
            att[i] = att[i].masked_fill(padding_pos[i] == 1, -myInf)
            att[i] = att[i].masked_fill(padding_pos[i].view(1, -1).transpose(0,1) == 1, -myInf)
        att = F.softmax(att, dim=-1)
        att = att @ multi_v   # batch_size * n_head * max_sequence_len * dim_qkv
        att = att.transpose(1, 2).contiguous().view(batch_size, self.config.max_sequence_len, self.config.n_head * self.dim_model)
        att = self.linear(att)
        return att

'''
encoder与decoder交互多头注意力

padding mask
'''
class MyMutualMultiAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.multi_q = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.multi_k = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.multi_v = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.linear = nn.Linear(config.dim_qkv * config.n_head, config.embed_size)
        self.dim_model = config.dim_qkv
        self.config = config
        
        '''init param'''
        torch.nn.init.normal_(self.multi_q.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.multi_k.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.multi_v.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    # x: batch_size * max_sequence_len * embed_size
    # encoder_input, decoder_input: batch_size * max_sequence_len * embed_size
    # encoder_padding_pos, decoder_padding_pos: batch_size * max_sequence_len
    def forward(self, encoder_input, decoder_input, encoder_padding_pos, decoder_padding_pos):
        '''q from decoder, kv from encoder'''
        batch_size = decoder_input.size(dim=0)
        multi_q = self.multi_q(decoder_input)   # batch_size * max_sequence_len * (n_head * dim_qkv)
        multi_k = self.multi_k(encoder_input)
        multi_v = self.multi_v(encoder_input)
        
        '''do reshape and transpose'''
        multi_q = multi_q.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.dim_model)
        multi_q = multi_q.transpose(1, 2)    # batch_size * n_head * max_sequence_len *  dim_qkv
        multi_k = multi_k.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.dim_model)
        multi_k = multi_k.transpose(1, 2)   # batch_size * n_head * max_sequence_len *  dim_qkv
        multi_v = multi_v.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.dim_model)
        multi_v = multi_v.transpose(1, 2)   # batch_size * n_head * max_sequence_len *  dim_qkv
        
        att = multi_q @ (multi_k.transpose(-1, -2))
        att = att / math.sqrt(self.dim_model)
        '''padding mask'''
        for i in range(batch_size):
            att[i] = att[i].masked_fill(encoder_padding_pos[i] == 1, -myInf)
            att[i] = att[i].masked_fill(decoder_padding_pos[i].view(1, -1).transpose(0, 1) == 1, -myInf)
        att = F.softmax(att, dim=-1)
        att = att @ multi_v
        att = att.transpose(1, 2).contiguous().view(batch_size, self.config.max_sequence_len, self.config.n_head * self.dim_model)
        att = self.linear(att)
        return att

'''
decoder self attention

padding mask and sequence mask
'''
class MyMaskedMultiAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.multi_q = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.multi_k = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.multi_v = nn.Linear(config.embed_size, config.dim_qkv * config.n_head)
        self.linear = nn.Linear(config.dim_qkv * config.n_head, config.embed_size)
        self.dim_model = config.dim_qkv
        self.config = config
        self.register_buffer('sequence_padding', torch.tril(torch.ones(self.config.max_sequence_len, self.config.max_sequence_len)))
        
        '''init param'''
        torch.nn.init.normal_(self.multi_q.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.multi_k.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.multi_v.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    # x: batch_size * max_sequence_len * embed_size
    # padding_pos: batch_size * max_sequence_len
    def forward(self, deocder_input, padding_pos):
        batch_size = deocder_input.size(dim=0)
        multi_q = self.multi_q(deocder_input)   # batch_size * max_sequence_len * (dim_qkv * n_head)
        multi_k = self.multi_k(deocder_input)
        multi_v = self.multi_v(deocder_input)
        
        '''do reshape and transpose'''
        multi_q = multi_q.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.dim_model)
        multi_q = multi_q.transpose(1, 2)   # batch_size * n_head * max_sequence_len * dim_qkv
        multi_k = multi_k.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.dim_model)
        multi_k = multi_k.transpose(1, 2)   # batch_size * n_head * max_sequence_len * dim_qkv
        multi_v = multi_v.view(batch_size, self.config.max_sequence_len, self.config.n_head, self.dim_model)
        multi_v = multi_v.transpose(1, 2)   # batch_size * n_head * max_sequence_len * dim_qkv
        
        att = multi_q @ (multi_k.transpose(-1, -2))
        att = att / math.sqrt(self.dim_model)
        '''padding mask'''
        for i in range(batch_size):
            att[i] = att[i].masked_fill(padding_pos[i] == 1, -myInf)
            att[i] = att[i].masked_fill(padding_pos[i].view(1, -1).transpose(0, 1) == 1, -myInf)
        '''sequence mask'''
        att = att.masked_fill(self.sequence_padding == 0, -myInf)
        
        att = F.softmax(att, dim=-1)
        att = att @ multi_v
        att = att.transpose(1, 2).contiguous().view(batch_size, self.config.max_sequence_len, self.config.n_head * self.dim_model)
        att = self.linear(att)
        return att

'''
前馈神经网络
'''
class MyFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.Linear(self.config.embed_size, self.config.embed_size * 4)
        self.ln_2 = nn.Linear(self.config.embed_size * 4, self.config.embed_size)
        self.relu = nn.ReLU()
        
        '''init param'''
        torch.nn.init.normal_(self.ln_1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.ln_2.weight, mean=0, std=0.01)

    # x: batch_size * max_sequence_len * embed_size
    def forward(self, x):
        x = self.ln_1(x)
        x = self.relu(x)
        x = self.ln_2(x)
        return x

'''
encoder中的block
'''
class MyEncoderBlock(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.multiAttention = MyMultiAttention(config)
        # self.dropout_1 = nn.Dropout(p=0.1)
        self.layerNorm_1 = nn.LayerNorm(self.config.embed_size)
        self.layerNorm_2 = nn.LayerNorm(self.config.embed_size)
        self.ffn = MyFFN(config=config)
        # self.dropout_2 = nn.Dropout(p=0.1)

    # x: batch_size * max_sequence_len * embed_size
    # padding_pos: batch_size * max_sequence_len
    def forward(self, x, padding_pos=None):
        attn = self.multiAttention(self.layerNorm_1(x), padding_pos)
        # attn = self.dropout_1(attn)
        x = attn + x
        # x = self.layerNorm_1(x)
        y = self.ffn(self.layerNorm_2(x))
        # y = self.dropout_2(y)
        x = y + x
        # x = self.layerNorm_2(x)
        return x

'''
encoder
'''
class MyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([MyEncoderBlock(config=config) for _ in range(self.config.N)])

    # encoder_input: batch_size * max_sequence_len * embed_size
    # encoder_padding_pos: batch_size * max_sequence_len
    def forward(self, encoder_input, encoder_padding_pos=None):
        i = 0
        for block in self.blocks:
            encoder_input= block(encoder_input, encoder_padding_pos)
            # print(f'after encoder block {i}, encoder_input = {encoder_input}')
            i += 1
        return encoder_input

'''
decoder中的block
'''
class MyDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.maskMultiAttention = MyMaskedMultiAttention(config=config)
        # self.dropout_1 = nn.Dropout(p=0.1)
        self.layerNorm_1 = nn.LayerNorm(config.embed_size)
        self.mutualMultiAttention = MyMutualMultiAttention(config=config)
        # self.dropout_2 = nn.Dropout(p=0.1)
        self.layerNorm_2 = nn.LayerNorm(config.embed_size)
        self.ffn = MyFFN(config=config)
        # self.dropout_3 = nn.Dropout(p=0.1)
        self.layerNorm_3 = nn.LayerNorm(config.embed_size)

    # encoder_input: batch_size * max_sequence_len * embed_size
    # decoder_input: batch_size * max_sequence_len * embed_size
    # encoder_padding_pos: batch_size * max_sequence_len
    # decoder_padding_pos: batch_size * max_sequence_len
    def forward(self, encoder_input, decoder_input, encoder_padding_pos=None, decoder_padding_pos=None):
        attn = self.maskMultiAttention(self.layerNorm_1(decoder_input), decoder_padding_pos)
        # attn = self.dropout_1(attn)
        decoder_input = decoder_input + attn
        # decoder_input = self.layerNorm_1(decoder_input)
        attn = self.mutualMultiAttention(encoder_input, self.layerNorm_2(decoder_input), encoder_padding_pos, decoder_padding_pos)
        # attn = self.dropout_2(attn)
        decoder_input = decoder_input + attn
        # decoder_input = self.layerNorm_2(decoder_input)
        y = self.ffn(self.layerNorm_3(decoder_input))
        # y = self.dropout_3(y)
        decoder_input = decoder_input + y
        # decoder_input = self.layerNorm_3(decoder_input)
        return decoder_input

'''
decoder
'''
class MyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([MyDecoderBlock(config=config) for _ in range(config.N)])
        self.linear = nn.Linear(config.embed_size, config.vocab_size)
    
    # encoder_input: batch_size * max_sequence_len * embed_size
    # encoder_padding_pos: batch_size * max_sequence_len
    def forward(self, encoder_input, decoder_input, encoder_padding_pos=None, decoder_padding_pos=None):
        '''debug'''
        i = 0
        for block in self.blocks:
            decoder_input = block(encoder_input, decoder_input, encoder_padding_pos, decoder_padding_pos)
            # print(f'after decoder block {i}, decoder_input = {decoder_input}')
            i += 1
        x = self.linear(decoder_input)
        # x = F.softmax(x, dim=-1) cross_entropy has softmax    # batch_size * max_sequence_len * vocab_size
        return x

'''
complete transformer
'''
class MyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.dim_qkv * config.n_head == config.embed_size, F"dim_qkv * n_head == embed_size!!!"
        self.embed = MyEmbedding(config=config)
        self.encoder = MyEncoder(config=config)
        self.decoder = MyDecoder(config=config)
        '''set embedding layer and projection after decoder with the same weights'''
        self.embed.input_embedding.weight = self.decoder.linear.weight

    # x: batch_size * max_sequence_len
    def forward(self, encoder_x, decoder_x, encoder_padding_pos, decoder_padding_pos, targets=None, is_training=True):
        '''embedding first'''
        encoder_x = self.embed(encoder_x)
        decoder_x = self.embed(decoder_x)
        # decoder_x: batch_size * max_sequence_len * embed_size
        encoder_x= self.encoder(encoder_x, encoder_padding_pos)
        
        model_output = self.decoder(encoder_x, decoder_x, encoder_padding_pos, decoder_padding_pos)
        # model_output: batch_size * max_sequence_len * vocab_size
        
        '''calculate loss'''
        loss = None
        if is_training:
            assert targets is not None, F'targets is none while training'
            # ignore label 0, which is padding
            loss = F.cross_entropy(model_output.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=0)
        return model_output, loss


