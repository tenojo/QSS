from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_from_disk
import torch
import numpy as np




def main():
    MODEL = "Qwen/Qwen3-4B-Thinking-2507"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    ds = load_from_disk("data/embs/remaining.hf")

    
    def tokenizeDS(e):
        return tokenizer(e['text'], return_tensors='pt')
    ds = ds.map(tokenizeDS)

    #Function that is slightly adjusted form the HuggingFace tutorial on calculating Perplexity
    def calcPPL(e):

        max_length = 32768
        stride = 512
        seq_len = e.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = e[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        return torch.exp(avg_nll)

    #Separate function so that we can use batch processing
    def mapPPLs(e):
        return {'PPL':[calcPPL(torch.from_numpy(np.array(x))) for x in e['input_ids']]}
    
    ds = ds.map(mapPPLs, batched=True)

    ds = ds.remove_columns(['input_ids', 'attention_mask'])

    ds.save_to_disk("data/embs/remaining_ppls.hf")


if __name__ == "__main__":
    main()