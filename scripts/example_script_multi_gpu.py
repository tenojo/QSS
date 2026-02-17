from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_from_disk
import sys
import os
os.environ['WANDB_MODE'] = 'disabled'
import torch
import numpy as np
from tqdm import tqdm



def main():

    MODEL = "Qwen/Qwen3-4B-Thinking-2507"

    ds = load_from_disk("data/embs/remaining.hf").select(range(25))

    #Separate function so that we can use batch processing
    #def mapPPLs(e):
    #    return {'PPL':[calcPPL(x) for x in e]}
    model = AutoModelForCausalLM.from_pretrained(MODEL , dtype=torch.bfloat16, tp_plan="auto")
    print(model._tp_plan)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    with tqdm(range(25), desc="Trying to get ppls...") as pbar:
        for ex in ds:
            prompt = ex['text']
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(inputs.input_ids.to(model.device), labels=inputs['input_ids'])
                loss = outputs.loss
                ppl = torch.exp(loss)
            ex['PPL'] = ppl
            pbar.update()

    print("Saving to disk...")

    ds.save_to_disk("data/embs/remaining_ppls.hf")

    sys.exit(0)


if __name__ == "__main__":
    main()