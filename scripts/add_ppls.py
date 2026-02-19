#!/usr/bin/env python3

import sys
import os
import torch

from math import exp
from statistics import mean
from logging import warning
from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm


def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('ds_path')
    return ap


def metrics(text, tokenizer, model):

    loss_fct = CrossEntropyLoss(reduction='sum')

    #losses, ppls, cppls = [], [], []
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors='pt'
    ).to(model.device)

    assert encoded.input_ids.shape[0] == 1
    if encoded.input_ids.shape[1] < 2:
            return (float(-1), float(-1), float(-1))

    with torch.no_grad():
        output = model(**encoded, labels=encoded.input_ids)

    shift_logits = output.logits[:, :-1, :]
    shift_labels = encoded.input_ids[:, 1:]

    batch_size, seq_length, vocab_size = shift_logits.shape
    assert batch_size == 1
    
    total_loss = loss_fct(
        shift_logits.view(batch_size * seq_length, vocab_size),
        shift_labels.view(batch_size * seq_length)
    )

    # if not torch.isclose(total_loss/seq_length, output.loss):
    #     warning(f'torch loss {float(total_loss/seq_length)} != '
    #             f'model loss {float(output.loss)}')
    # Added some error prevention... There is apparently some case where seq_length is 0 -> division by 0????
    # Will investigate at a later date how this is possible...
    if seq_length == 0:
        return (float(-1), float(-1), float(-1))

    loss = float(total_loss) / seq_length
    char_length = len(tokenizer.decode(shift_labels[0]))

    #losses.append(loss)
    #ppls.append(exp(loss))
    #cppls.append(exp(total_loss/char_length))

    return (loss, exp(loss), exp(total_loss/char_length))


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        #device_map='auto',
        tp_plan='auto',
        dtype=torch.bfloat16,
    )

    ds = load_from_disk(args.ds_path)
    losses, ppls, cppls = [], [], []
    with tqdm(range(len(ds)), desc="Calculating ppls...") as pbar:
        print('\n')
        for ex in ds:
            try:
                loss, ppl, cppl = metrics(ex['text'], tokenizer, model)
                #print(f'Perplexity is {ppl}')
                #print(f'{bn} mean loss: {loss:.2f}')
                #print(f'{bn} mean ppl : {ppl:.2f}')
                #print(f'{bn} mean cppl: {cppl:.2f}')
                losses.append(loss)
                ppls.append(ppl)
                cppls.append(cppl)
            except:
                losses.append(float(-1))
                ppls.append(float(-1))
                cppls.append(float(-1))
            pbar.update(1)
    
    ds = ds.add_column('loss', losses)
    ds = ds.add_column('ppl', ppls)
    ds = ds.add_column('cppl', cppls)
    
    ds.save_to_disk(args.ds_path.replace(".hf", "_ppls.hf"))
    #print(f'overall mean loss: {mean(losses):.2f}')
    #print(f'overall mean ppl : {mean(ppls):.2f}')
    #print(f'overall mean cppl: {mean(cppls):.2f}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
