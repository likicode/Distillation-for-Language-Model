from model.LSTM_LM import RNNModel

import random
import time
import numpy as np
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger
from pytorch_pretrained_bert import TransfoXLLMHeadModel, TransfoXLCorpus, TransfoXLTokenizer


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/penn',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ptb',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'ptb'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=80,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
parser.add_argument('--finetuned', action='store_true',
                    help='use finetuned model')
parser.add_argument('--augment_train', action='store_true', help='use augmented train data')
parser.add_argument('--finetuned_model', type=str, default='ptb_finetuned_model/',
                    help='use finedtuned Transformer-XL model on PTB dataset')
parser.add_argument('--model_name', type=str, default='ptb_model/transformerxl/',
                    help='pretrained Transformer-XL provided by https://github.com/huggingface/pytorch-pretrained-BERT')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
# LSTM model related params
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--temperature', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--distill_loss_scale', type=float, default=0.1,
                    help='the scale of BCE loss')
parser.add_argument('--save', type=str, default='distilled_model/model.pt',
                    help='path to save the final model')
parser.add_argument('--log_dir', type=str, default='distilled_log/',
                    help='path to save the log file')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--top_k', type=int, default=3,
                    help='top k largest of teacher model logit output')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
args = parser.parse_args()

assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
torch.cuda.set_device(args.device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device != -1:
    torch.cuda.manual_seed(args.seed)
timestamp = str(int(time.time())) + "_"
fname = timestamp + "dis_scale_" + str(args.distill_loss_scale) + "epochs_" + str(args.epochs) + "top_k_" + str(args.top_k)
logging = get_logger(os.path.join(args.log_dir, fname+'log.txt'), log_=not args.no_log)


##############
# Load dataset
##############
corpus = TransfoXLCorpus.from_pretrained(args.model_name)
ntokens = len(corpus.vocab)
tokenizer = TransfoXLTokenizer.from_pretrained(args.model_name)

if args.finetuned:
    if args.augment_train:
        corpus.train = tokenizer.encode_file(
            os.path.join(args.data, 'train_aug.txt'), ordered=True)
    else:
        corpus.train = tokenizer.encode_file(
            os.path.join(args.data, 'train.txt'), ordered=True)
    corpus.valid = tokenizer.encode_file(
        os.path.join(args.data, 'valid.txt'), ordered=True)
    corpus.test = tokenizer.encode_file(
        os.path.join(args.data, 'test.txt'), ordered=True)
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
else:
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
print("finish load dataset")

################
# Load Model
################
if args.finetuned:
    # Load finetuned model
    with open(os.path.join(args.finetuned_model, 'model.pt'), 'rb') as f:
        model = torch.load(f, map_location='cuda:0')
else:
    # Load Huggingface pre-trained model
    model = TransfoXLLMHeadModel.from_pretrained(args.model_name)

model = model.to(device)
print("finish loading model")

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

################################
# Build distilled LSTM model
################################
distilled_model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
distilled_model = distilled_model.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(test_iter):
    distilled_model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    hidden = distilled_model.init_hidden(args.batch_size)
    criterion_perplexity = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, (data, target, seq_len) in enumerate(test_iter):
            data = data[:, -seq_len:]
            data = data.transpose(1, 0).contiguous()
            target = target.transpose(1, 0).contiguous()
            output, hidden = distilled_model(data, hidden)
            loss = criterion_perplexity(output.view(-1, ntokens), target.view(-1))
            total_loss += seq_len * loss.item()
            total_len += seq_len
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
        total_time, 1000 * total_time / (idx + 1)))
    return total_loss / total_len



def train(train_iter):
    distilled_model.train()
    model.eval()
    loss_ce_global = 0.
    loss_kd_global = 0.
    start_time = time.time()
    total_len = 0
    hidden = distilled_model.init_hidden(args.batch_size)

    #criterion_distill = nn.KLDivLoss(reduction='batchmean')
    criterion_distill = nn.BCEWithLogitsLoss()
    criterion_perplexity = nn.CrossEntropyLoss()
    optimizer = Adam(distilled_model.parameters(), args.lr)

    train_step = 0
    for idx, (data, target, seq_len) in enumerate(train_iter):
        data = data[:, -seq_len:]

        """With huggingface Pretrained Model"""

        with torch.no_grad():
            new_mems = None
            bsz, _ = data.size()
            t_hid, new_mems = model.transformer(data, new_mems)
            pred_hid = t_hid[:, -seq_len:]
            t_hidden = pred_hid.contiguous().view(-1, pred_hid.size(-1))
            t_output = model.crit.forward(t_hidden)
            t_output = t_output.view(bsz, seq_len, ntokens)
            total_len += seq_len

        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        data = data.transpose(1, 0).contiguous()
        target = target.transpose(1, 0).contiguous()
        s_output, hidden = distilled_model(data, hidden)
        s_output = s_output.view(bsz, seq_len, ntokens)
        ce_loss = criterion_perplexity(s_output.view(-1, ntokens), target.view(-1))
        loss_ce_global += ce_loss.item()

        t_output = torch.sigmoid(t_output)

        # select the topk for each word in t_output, then only retrieve the corresponding prob in s_output
        _, top_idx = t_output.topk(k=args.top_k, dim=-1)
        m_idx = torch.zeros_like(t_output).scatter_(2, top_idx, 1).byte()
        t_output_masked = t_output.masked_select(m_idx).view(-1, args.top_k)
        s_output_masked = s_output.masked_select(m_idx).view(-1, args.top_k)
        kd_loss = criterion_distill(s_output_masked, t_output_masked)
        loss_kd_global += kd_loss.item()

        (args.distill_loss_scale * kd_loss + (1-args.distill_loss_scale) * ce_loss).backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        train_step += 1

        if train_step % args.log_interval == 0:
            cur_loss = loss_ce_global / args.log_interval
            cur_kd_loss = loss_kd_global / args.log_interval
            elapsed = time.time() - start_time
            log_str = '| epoch {:3d} step {:>8d} | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | ce loss {:5.4f}| kd loss {:5.4f}| perplexity {:9.3f}'.format(
                epoch, train_step, lr, elapsed * 1000 / args.log_interval, cur_loss, cur_kd_loss, math.exp(cur_loss))
            logging(log_str)
            loss_ce_global = 0
            loss_kd_global = 0
            start_time = time.time()



# Loop over epochs.
lr = args.lr
best_val_loss = None
save_file = args.save.split(".")[0] + timestamp + fname + "_bce.pt"

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(tr_iter)
        val_loss = evaluate(va_iter)
        logging('-' * 89)
        logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.2f}| valid ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        logging('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_file, 'wb') as f:
                torch.save(distilled_model, f)
            best_val_loss = val_loss

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


# Load the best saved model.
with open(save_file, 'rb') as f:
    distilled_model = torch.load(f)

    distilled_model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(te_iter)
logging('=' * 89)
logging('| End of training |test loss: {:5.2f} |test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging('=' * 89)








