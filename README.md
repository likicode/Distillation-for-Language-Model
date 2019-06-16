###Distillation for Language Model

This repository distills the Transformer-XL model into a simple LSTM model.

Overview

* `finetune-txl.py` - Finetune pre-trained Transformer-XL model provided by https://github.com/huggingface/pytorch-pretrained-BERT on PTB dataset.

```
| end of epoch   1 | time: 2078.10s | valid loss:  3.59| valid ppl    36.19
| End of training | test loss  3.55 | test ppl    34.820
```

* `data-augment-txl.py` - keep the first *context_length* tokens unchanged, use fine-tuned Transformer-XL model to predict the following tokens. The *context_length* is set to 40 and 60 in this experiment.
* `distill.py` - Distill the fine-tuned Transformer-XL into a simple LSTM model, with the augmented dataset

```
python distill.py --finetuned --augment_train --top_k 3 --distill_loss_scale 0.1
| End of training | test ppl    68.84

```

Reference

* Dai, Zihang, et al. "Transformer-xl: Attentive language models beyond a fixed-length context." arXiv preprint arXiv:1901.02860 (2019).
* **PyTorch Pretrained BERT**: https://github.com/huggingface/pytorch-pretrained-BERT
* **Word-level language modeling RNN**: https://github.com/pytorch/examples/tree/master/word_language_model


