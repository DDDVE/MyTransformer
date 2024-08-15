import tiktoken
import torch
import random
import sys

'''read local file and split into batches'''
class data_loader:
    def __init__(self, batch_size=64, max_sequence_len=1024, test_rate=0.8) -> None:
        '''read by line and move '\n' '''
        with open('./cn.txt', 'r', encoding='utf-8') as f:
            self.cn_total = f.readlines()
            for i in range(len(self.cn_total)):
                if self.cn_total[i][-1] == '\n':
                    self.cn_total[i] = self.cn_total[i][:-1]
        f.close()
        with open('./en.txt', 'r', encoding='utf-8') as f:
            self.en_total = f.readlines()
            for i in range(len(self.en_total)):
                if self.en_total[i][-1] == '\n':
                    self.en_total[i] = self.en_total[i][:-1]
        f.close()
            
        # # so far cn_total and en_total have space between each word, now we squeeze them
        # for i in range(len(self.cn_total)):
        #     cn_a, en_a = self.cn_total[i], self.en_total[i]
        #     cn_a, en_a = cn_a.split(), en_a.split()
        #     cn_b, en_b = ''.join(cn_a), ''.join(en_a)
        #     self.cn_total[i], self.en_total[i] = cn_b, en_b
        
        # so far each sentence in cn_total has no start-end symbol, now we add them
        for i in range(len(self.cn_total)):
            self.cn_total[i] = '< ' + self.cn_total[i] + ' >'
            self.en_total[i] = '< ' + self.en_total[i] + ' >'
        '''now we do tokenization and total train-test set'''
        self.encoder_input, self.decoder_input, self.targets = [], [], []
        tik_encoder = tiktoken.get_encoding('gpt2')
        for i in range(len(self.en_total)):
            a = tik_encoder.encode(self.en_total[i])
            self.encoder_input.append(a)
            b = tik_encoder.encode(self.cn_total[i][ : -1]) # exclude 'E'
            self.decoder_input.append(b)
            c = tik_encoder.encode(self.cn_total[i][1:])    # exclude 'S'
            self.targets.append(c)
        self.batch_size = batch_size
        self.max_sequence_len=max_sequence_len
        self.pos = 0    # index of reading data batch
        self.test_rate = test_rate
        self.left_bracket = tik_encoder.encode('< a')[0]
        self.S = tik_encoder.encode('S')[0]
        self.E = tik_encoder.encode('E')[0]
        self.right_bracket = tik_encoder.encode('a >')[-1]
    
    '''
    return: tensor
    '''
    def get_next_batch(self):
        encoder_list = self.encoder_input[self.pos : self.pos + self.batch_size]
        decoder_list = self.decoder_input[self.pos : self.pos + self.batch_size]
        targets_list = self.targets[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        if (self.pos + self.batch_size > len(self.encoder_input) * self.test_rate):
            self.pos = 0
            
        '''make tensor and padding'''
        return self.get_useful(encoder_list, decoder_list, targets_list)
        
        
    def get_useful(self, encoder_list, decoder_list, targets_list):
        encoder_input = torch.zeros((self.batch_size, self.max_sequence_len))
        decoder_input = torch.zeros((self.batch_size, self.max_sequence_len))
        targets = torch.zeros((self.batch_size, self.max_sequence_len))
        encoder_padding_pos = torch.zeros((self.batch_size, self.max_sequence_len))
        decoder_padding_pos = torch.zeros((self.batch_size, self.max_sequence_len))
        # encoder
        for i in range(self.batch_size):
            for j in range(min(len(encoder_list[i]), self.max_sequence_len)):
                encoder_input[i, j] = encoder_list[i][j]
            '''add <E>'''
            last_pos = min(len(encoder_list[i]), self.max_sequence_len)
            encoder_input[i, last_pos - 1] = self.right_bracket
            # encoder_input[i, last_pos - 2] = self.E
            # encoder_input[i, last_pos - 3] = self.left_bracket
            for j in range(len(encoder_list[i]), self.max_sequence_len):
                encoder_padding_pos[i][j] = 1
        # decoder
        for i in range(self.batch_size):
            for j in range(min(len(decoder_list[i]), self.max_sequence_len)):
                decoder_input[i, j] = decoder_list[i][j]
            for j in range(len(decoder_list[i]), self.max_sequence_len):
                decoder_padding_pos[i][j] = 1
        # targets
        for i in range(self.batch_size):
            for j in range(min(len(targets_list[i]), self.max_sequence_len)):
                targets[i, j] = targets_list[i][j]
            '''add <E>'''
            last_pos = min(len(targets_list[i]), self.max_sequence_len)
            targets[i, last_pos - 1] = self.right_bracket
            # targets[i, last_pos - 2] = self.E
            # targets[i, last_pos - 3] = self.left_bracket
        return encoder_input.long(), decoder_input.long(), encoder_padding_pos.long(), decoder_padding_pos.long(), targets.long()
    
    def get_eval_batch(self):
        start = random.randint(int(len(self.encoder_input) * self.test_rate), len(self.encoder_input) - self.batch_size)
        encoder_list = self.encoder_input[start : start + self.batch_size]
        decoder_list = self.decoder_input[start : start + self.batch_size]
        targets_list = self.targets[start : start + self.batch_size]
        '''make tensor and padding'''
        return self.get_useful(encoder_list, decoder_list, targets_list)
            
            
        