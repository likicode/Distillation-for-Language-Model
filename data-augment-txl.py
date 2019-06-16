import argparse
import logging
import os
from io import open

import numpy as np
import torch
from pytorch_pretrained_bert import TransfoXLCorpus, TransfoXLTokenizer
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--max_ctx_len', type=int, default=80, help='')
parser.add_argument('--max_gen_len', type=int, default=512, help='')
parser.add_argument('--topk', type=int, default=40)
parser.add_argument('--start_idx', type=int, default=-1, help='')
parser.add_argument('--out_path', type=str, default='output.txt', help='')
parser.add_argument('--finetuned_model', type=str, default='ptb_finetuned_model/',
                    help='use finedtuned Transformer-XL model on PTB dataset')
parser.add_argument('--model_name', type=str, default='ptb_model/transformerxl/',
                    help='pretrained Transformer-XL provided by https://github.com/huggingface/pytorch-pretrained-BERT')
parser.add_argument('--data', type=str, default='../data/penn',
                    help='location of the data corpus')
parser.add_argument('--context_length', type=int, default=60, help='keep these tokens unchanged')
parser.add_argument('--tgt_len', type=int, default=80,
                    help='total number of output tokens')
parser.add_argument('--outfn', type=str, default='LM_aug_data/penn.txt')
parser.add_argument('--batch_size', type=int, default=15,
                    help='batch size')
parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
torch.cuda.set_device(args.device)

##############
# Load dataset
##############
corpus = TransfoXLCorpus.from_pretrained(args.model_name)
ntokens = len(corpus.vocab)
tokenizer = TransfoXLTokenizer.from_pretrained(args.model_name)
corpus.train = tokenizer.encode_file(os.path.join(args.data, 'train.txt'), ordered=True)
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                              device=device, ext_len=0)

for idx, sym in enumerate(tokenizer.idx2sym):
    tokenizer.idx2sym[idx] = sym.encode('latin1').decode('utf-8')


def format_text(tokens):
    line = ''
    for token in tokens:
        if token == '<eos>':
            line += '\n'
        else:
            line += token
            line += ' '

    # simple rules of detokenization
    line = line.replace(' @-@ ', '-')
    line = line.replace(' @,@ ', ',')
    line = line.replace(' @.@ ', '.')
    line = line.replace(' . ', '. ')
    line = line.replace(' , ', ', ')
    line = line.replace(' : ', ': ')
    line = line.replace(' ; ', '; ')
    line = line.replace(" 's ", "'s ")
    line = line.replace(' ( ', ' (')
    line = line.replace(' ) ', ') ')

    return line


# Load pre-trained model (weights)
with open(os.path.join(args.finetuned_model, 'model.pt'), 'rb') as f:
    model = torch.load(f, map_location='cuda:0')
model = model.to(device)
model.eval()

unk_id = tokenizer.convert_tokens_to_ids(['<unk>'])[0]
generate_len = args.tgt_len - args.context_length

generation = defaultdict(list)
outf = open(args.outfn, 'a+')
with torch.no_grad():
    for idx, (data, target, seq_len) in tqdm(enumerate(tr_iter)):
        if idx % 50 == 0:
            print(idx)
        data = data[:, -seq_len:]
        tensor_source = data[:args.context_length]
        for k in range(args.batch_size):
            generation[k] = []
        for i in range(generate_len):
            if i == 0:
                log_prob, mems = model(tensor_source)
            else:
                log_prob, mems = model(tensor_source, mems=mems)

            prob = torch.exp(log_prob[:, -1, :])
            prob[:, unk_id].data.fill_(0.)

            # sample from the top-k tokens
            top_prob, top_index = torch.topk(prob, args.topk)
            token_idx = torch.multinomial(top_prob, 1)
            token = top_index.gather(1, token_idx)
            tensor = token.detach()
            symbols = token.cpu().flatten().numpy()
            for m in range(args.batch_size):
                generation[m].append(tokenizer.get_sym(symbols[m]))
        for p in range(args.batch_size):
            outf.write(format_text(tokenizer.convert_ids_to_tokens(tensor_source[p])))
            outf.write(format_text(generation[p]) + '\n')

