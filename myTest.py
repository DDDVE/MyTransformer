import torch
import tiktoken
# import my_transformer
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def try_real_sentence(max_sequence_len, model):
    # print(f'{max_sequence_len=}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    en = '< on the two rest days of the week , when things are quietest , liu mingshan brings flowers and stands alone before his wife \'s memorial . >'
    assert len(en) < max_sequence_len, F'sentence len cannot beyond {max_sequence_len}'
    cn = "< "    # start char
    
    encoder_input = torch.zeros((1, max_sequence_len))
    encoder_padding_pos = torch.zeros((1, max_sequence_len))
    decoder_input = torch.zeros((1, max_sequence_len))
    decoder_padding_pos = torch.zeros((1, max_sequence_len))
    tik_encoder = tiktoken.get_encoding('gpt2')
    
    en = tik_encoder.encode(en) # list
    cn = tik_encoder.encode(cn)
    end_char_1 = tik_encoder.encode('>')[-1]
    end_char_2 = tik_encoder.encode('a >')[-1]
    last_padding_pos = len(cn)
    
    '''make tensor and padding pos'''
    # encoder
    for i in range(len(en)):
        encoder_input[0, i] = en[i]
    for i in range(len(en), max_sequence_len):
        encoder_padding_pos[0, i] = 1
    # decoder
    for i in range(len(cn)):
        decoder_input[0, i] = cn[i]
    for i in range(len(cn), max_sequence_len):
        decoder_padding_pos[0, i] = 1
        
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    encoder_padding_pos = encoder_padding_pos.to(device)
    decoder_padding_pos = decoder_padding_pos.to(device)
    
    while True:
        output, _ = model(encoder_input.long(), decoder_input.long(), encoder_padding_pos.long(), decoder_padding_pos.long(), None, False)
        # output: 1 * max_sequence_len * vocab_size
        _, next = torch.max(torch.nn.functional.softmax(output[0, last_padding_pos-1], dim=-1), dim=-1)
        print(f'return sequence len = {last_padding_pos} | next char is {next}')
        # modify decoder_input and padding
        decoder_input[0, last_padding_pos] = next.item()
        decoder_padding_pos[0, last_padding_pos] = 0
        last_padding_pos += 1
        if next.item() == end_char_1 or next.item() == end_char_2 or last_padding_pos >= max_sequence_len:
            break
    cn = tik_encoder.decode(decoder_input[0, :last_padding_pos].int().tolist())
    print(f'{cn=}')
    return cn

'''load model'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('train/2024_8_8_22_37/45000.pth', map_location=torch.device(device))
try_real_sentence(model.config.max_sequence_len, model)
    