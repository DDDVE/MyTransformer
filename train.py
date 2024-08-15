import data_loader
import my_transformer
import torch
import time
import sys
import os
import tiktoken
import math
    
prog_start = time.time()

# ***************************************check train folder and make today's record folder*********************************
train_folder = './train'
if not os.path.exists(train_folder):
    print('local folder has no train folder, make it')
    os.mkdir(train_folder)
today = time.localtime()
today_folder = ''.join([str(today.tm_year), '_', 
                        str(today.tm_mon), '_', str(today.tm_mday), '_', str(today.tm_hour), '_', str(today.tm_min)])
today_folder = '/'.join([train_folder, today_folder])
if not os.path.exists(today_folder):
    print(f'today\'s folder {today_folder} not exists, make it')
    os.mkdir(today_folder)
    
'''make loss file, evaluate file, sentence file'''
loss_file_ptr = open(today_folder + '/loss.txt', 'w')
eva_file_ptr = open(today_folder + '/eval.txt', 'w')
sent_file_ptr = open(today_folder + '/sent.txt', 'w')
# *************************************************************************************************************************

# ***********************************************model config and data set*************************************************
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

config = my_transformer.MyTransformerConfig()
config.dim_qkv = int(config.embed_size / config.n_head)
config.max_sequence_len = 160

dataLoader = data_loader.data_loader(batch_size=16, max_sequence_len=config.max_sequence_len)

# print('out')
# sys.exit(0)

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 700
max_step = 20000
max_train_step = 100000

model = my_transformer.MyTransformer(config)

model = model.to(device)
# *************************************************************************************************************************

optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)

loss_track = []
eval_track = []

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_step:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_step - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

for step in range(max_train_step):
    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    '''get data set'''
    encoder_input, decoder_input, encoder_padding_pos, decoder_padding_pos, targets = dataLoader.get_next_batch()
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    encoder_padding_pos = encoder_padding_pos.to(device)
    decoder_padding_pos = decoder_padding_pos.to(device)
    targets = targets.to(device)
    
    output, loss = model(encoder_input, decoder_input, encoder_padding_pos, decoder_padding_pos, targets)
    loss.backward()
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    t1 = time.time()
    print(f'step = {step:7d} | loss = {loss:.4f} | time = {(t1-t0):.4f} | lr = {lr:.8f}')
    # if step > 5000:
    #     print(f'model output {output[:5, :, :]}')
    # *********************************************write into file**********************************************
    '''loss file format: step \t loss'''
    loss_file_ptr.write(f'{step:7d}\t\t{loss:.4f}')
    # **************************************************evaluate************************************************
    if step % 500 == 0:
        model.eval()
        encoder_input, decoder_input, encoder_padding_pos, decoder_padding_pos, targets = dataLoader.get_eval_batch()
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        encoder_padding_pos = encoder_padding_pos.to(device)
        decoder_padding_pos = decoder_padding_pos.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            _, loss = model(encoder_input, decoder_input, encoder_padding_pos, decoder_padding_pos, targets)
            eva_file_ptr.write(f'{step:7d}\t\t{loss:.4f}')
            print(f'eval loss:{loss:.4f}')

            '''try real sentence'''
            # sent_file_ptr.write(f'{step:7d}\t\t{myTest.try_real_sentence(config.max_sequence_len, model)}')
    # ************************************************save model**********************************************************
    if  step > 0 and step % 5000 == 0:
        # model file name: train/{step}.pth
        model_file_name = today_folder + '/' + f'{step}.pth'
        torch.save(model, model_file_name)
    
    
# close file ptr
loss_file_ptr.close()
eva_file_ptr.close()
sent_file_ptr.close()

prog_end = time.time()
print(f'training over, use {prog_end - prog_start} in total')



