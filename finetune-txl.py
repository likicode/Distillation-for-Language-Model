import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import time
import torch.optim as optim
from pytorch_pretrained_bert import TransfoXLLMHeadModel, TransfoXLCorpus, TransfoXLTokenizer


parser = argparse.ArgumentParser(description='Fine-tune Transformer-XL')
parser.add_argument('--model_name', type=str, default='ptb_model/transformerxl/',
                    help='pretrained Transformer-XL provided by https://github.com/huggingface/pytorch-pretrained-BERT')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--data', type=str, default='../data/penn',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ptb',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'ptb'],
                    help='dataset name')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=50,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=1600,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=1000,
                        help='max positional embedding index')
parser.add_argument('--same_length', action='store_true',
                        help='set same length attention with masking')
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--split', type=str, default='test',
                        choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
parser.add_argument('--log-interval', type=int, default=100,
                    help='report interval')
parser.add_argument('--save_model', type=str, default='ptb_finetuned_model/model.pt')
parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
torch.cuda.set_device(args.device)

corpus = TransfoXLCorpus.from_pretrained(args.model_name)
ntokens = len(corpus.vocab)

tokenizer = TransfoXLTokenizer.from_pretrained(args.model_name)


# get data
corpus.train = tokenizer.encode_file(
    os.path.join(args.data, 'train.txt'), ordered=True)
corpus.valid = tokenizer.encode_file(
    os.path.join(args.data, 'valid.txt'), ordered=True)
corpus.test = tokenizer.encode_file(
    os.path.join(args.data, 'test.txt'), ordered=True)

tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.tgt_len)
va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)
print("finish load dataset")


# Load a pre-trained model
model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
model = model.to(device)

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True
print("finish loading model")

###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    with torch.no_grad():
        mems = None
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model(data, target, mems)
            loss, mems = ret
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
        total_time = time.time() - start_time
    print('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len



def train():
    train_step = 0
    train_loss = 0
    model.train()
    mems = tuple()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eval_start_time = time.time()
    log_start_time = time.time()
    for batch, (data, target, seq_len) in enumerate(tr_iter):
        data = data[:, -seq_len:]
        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        train_loss += loss.float().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
                                   elapsed * 1000 / args.log_interval, cur_loss)
            log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            print(log_str)
            train_loss = 0
            log_start_time = time.time()


best_val_loss = None

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(va_iter)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.2f}| valid ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save_model, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


with open(args.save_model, 'rb') as f:
    model = torch.load(f)
model = model.to(device)

test_loss = evaluate(te_iter)
print('=' * 100)
print('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
        test_loss, math.exp(test_loss)))
print('=' * 100)












