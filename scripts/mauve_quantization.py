# A lot of this code is created or inspired by the MAUVE team (Pillutla et al. 2021)


#Imports
import os
os.environ['HF_HOME'] = '/scratch/project_2000539/tapio/HF_cache/'
import torch
from pprint import pprint
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import time
import faiss
from tqdm import tqdm



#Functions

def featurize_tokens_from_model(model, tokenized_texts, batch_size, name="", verbose=False):
    """Featurize tokenized texts using models, support batchify
    :param model: HF Transformers model
    :param batch_size: Batch size used during forward pass
    :param tokenized_texts: list of torch.LongTensor of shape (1, length)
    :param verbose: If True, print status and time
    :return:
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        t1 = time.time()
        feats, chunks, chunk_sent_lengths = [], [], []
        chunk_idx = 0

        while chunk_idx * batch_size < len(tokenized_texts):
            _chunk = [_t.view(-1) for _t in tokenized_texts[chunk_idx * batch_size: (chunk_idx + 1) * batch_size]]
            chunks.append(_chunk)
            chunk_sent_lengths.append([len(_c) for _c in _chunk])
            chunk_idx += 1

        for chunk, chunk_sent_length in tqdm(list(zip(chunks, chunk_sent_lengths)), desc=f"Featurizing {name}"):
            padded_chunk = torch.nn.utils.rnn.pad_sequence(chunk,
                                                        batch_first=True,
                                                        padding_value=0).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.ones(sent_length).long() for sent_length in chunk_sent_length],
                batch_first=True,
                padding_value=0).to(device)
            outs = model(input_ids=padded_chunk,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        output_hidden_states=True,
                        return_dict=True)
            h = []
            for hidden_state, sent_length in zip(outs.hidden_states[-1], chunk_sent_length):
                h.append(hidden_state[sent_length - 1])
            h = torch.stack(h, dim=0)
            feats.append(h.cpu())
        t2 = time.time()
        if verbose:
            print(f'Featurize time: {round(t2-t1, 2)}')
        return feats
    

# ## MAUVE style clustering of embeddings for generating distributions


#Cluster Distributions of Embeddings (CDOE)
#Modified version of the quanitzation procedure used in MAUVE (Pillutla et al. 2021, 2023)
def CDOE(p, q, num_clusters,
                  norm='none', whiten=True,
                  pca_max_data=-1,
                  explained_variance=0.9,
                  num_redo=5, max_iter=500,
                  seed=0, verbose=False):
    assert 0 < explained_variance < 1
    def _normalize(array):
        # Normalize sum of array to 1.
        # We assume non-negative entries with non-zero sum.
        return array / array.sum()
    if verbose:
        print(f'seed = {seed}')
    assert norm in ['none', 'l2', 'l1', None]
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    #PCA on embeddings
    pca = PCA(n_components=None, whiten=whiten, random_state=seed+1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed+5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(f'Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints')
    #Only enough components to retain given amount of variance (default 0.9)
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    if verbose:
        print(f'performing clustering in lower dimension = {idx}')
        pprint(f'Shape of embeddings before PCA transformation {data1.shape}')
    data1 = pca.transform(data1)[:, :idx+1]
    if verbose:
        pprint(f'Shape of embeddings after PCA transformation {data1.shape}')
    # Cluster features and obtain the labels for each data point.
    data1 = data1.astype(np.float32)  # Faiss requires float32.
    t1 = time.time()
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          verbose=verbose, nredo=num_redo, update_index=True,
                          seed=seed+2)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    t2 = time.time()
    if verbose:
        print('kmeans time:', round(t2-t1, 2), 's')

    #Following lines not part of the original MAUVE quantization :)
    q_labels = labels[:len(q)]
    #To hasten future calculations, zip the indices of all source texts with their respective FAISS-clusters (aka labels)
    q2cluster = dict(zip(range(len(q)), q_labels))
    p_labels = labels[len(q):]
    #Also noting down the reference-cluster_label data, so that we can probe whether the clusters remain the 'same' for two different runs
    p2cluster = dict(zip(range(len(p)), p_labels))

    #Origina code continues form here :)
    # Convert cluster labels to histograms.
    p_bin_counts = np.histogram(
        p_labels, bins=num_clusters,
        range=[0, num_clusters], density=False
    )[0]
    q_bin_counts = np.histogram(
        q_labels, bins=num_clusters,
        range=[0, num_clusters], density=False
    )[0]
    # Histograms without smoothing (used for the original MAUVE).
    p_hist = _normalize(p_bin_counts)
    q_hist = _normalize(q_bin_counts)
    # Histograms with Krichevsky-Trofimov smoothing.
    # Used for MAUVE* suggested by by Pillutla et al. (JMLR 2023).
    p_hist_smoothed = _normalize(p_bin_counts + 0.5)
    #Originally returns values as they are and not a dict
    return {'p_bin_counts':p_bin_counts, 'q_bin_counts':q_bin_counts, 'p_hist':p_hist, 'q_hist':q_hist, 'p_hist_smoothed':p_hist_smoothed, 'q2cluster':q2cluster, 'p2cluster':p2cluster}