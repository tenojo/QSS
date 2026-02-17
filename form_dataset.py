#Imports
from scripts import dataset_scripts as ds_s, mauve_quantization as mq
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset, load_from_disk
import numpy as np
import sys
from tqdm import tqdm

def main(cmd_args):
    # Dicts are in format: {ds_path, ds_name, under_ds_name}
    human_ds_dict = {"ds_path":"data/human/news-fi-2019.jsonl", "ds_name":"news-fi-2019.jsonl", "under_ds_name":None}
    clums_ds_dict = {"ds_path":"data/clumsified/news-fi-2019.jsonl_regeneration_5_mini_regen_round_1.jsonl", "ds_name":"news-fi-2019.jsonl_regeneration_5_mini_regen_round_1.jsonl", "under_ds_name":"news-fi-2019.jsonl"}


    ds = ds_s.format_datasets([human_ds_dict, clums_ds_dict])

    print("Dataset formatted!")

    emb_model = cmd_args[0]
    output_path = cmd_args[1]

    tokenizer = AutoTokenizer.from_pretrained(emb_model)
    model = AutoModel.from_pretrained(emb_model, tp_plan="auto")
    print("Tokenizing texts...\n")
    inputs = [tokenizer.encode(ds[i]['text'], return_tensors="pt", truncation=True, max_length=512) for i in tqdm(range(len(ds)))]
    print("Moving to featurizing...\n")
    embeddings = mq.featurize_tokens_from_model(model, inputs, 1, name="", verbose=False)
    inputs= []
    del inputs

    for i in range(len(ds)):
        ds[i]['embedding']= embeddings[i]

    ds_test = Dataset.from_list(ds)
    ds_test.save_to_disk(output_path)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))