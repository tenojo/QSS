import numpy as np
from scipy.spatial.distance import jensenshannon as jsd

#Function for getting the target n for each cluster
def get_target_n_per_cluster(p_clusters, q_clusters, subset_size, simple=False):
    simple_ratios = p_clusters/np.sum(p_clusters)
    simple_target_n = np.ceil(simple_ratios*subset_size)
    #If we want efficiency, then the target n for each cluster is just the smaller value between the number of q samples in cluster x,
    #Or the simple target amount based on the ratios of p samples per each cluster
    simple_q_cluster_n = {i:min(int(q_clusters[i]),int(x)) for i,x in enumerate(simple_target_n)}
    simple_q_c_total = np.sum(list(simple_q_cluster_n.values()))
    if simple or subset_size == simple_q_c_total:
        return simple_q_cluster_n
    #If we want an exact subset size, then either eliminate or add samples to minimize JSD between p and target q distributions
    if simple_q_c_total < subset_size:
        remaining_q_samples = [q_clusters[i]-simple_q_cluster_n[i] for i in range(len(q_clusters))]
        while simple_q_c_total < subset_size:
            possible_additions = [i for i in range(len(remaining_q_samples)) if remaining_q_samples[i] > 0]
            best = possible_additions[0]
            best_jsd = np.inf
            for x in possible_additions:
                temp = simple_q_cluster_n.copy()
                temp[x] += 1
                distance = jsd(p_clusters, np.array(list(temp.values())))
                if distance < best_jsd:
                    best = x
                    best_jsd = distance
            simple_q_cluster_n[best] += 1
            remaining_q_samples[best] += -1
            simple_q_c_total += 1
    #If there are too many samples in the simple distribution
    else:
        while simple_q_c_total > subset_size:
            possible_deletions = [i for i in simple_q_cluster_n if simple_q_cluster_n[i] > 0]
            best = possible_deletions[0]
            best_jsd = np.inf
            for x in possible_deletions:
                temp = simple_q_cluster_n.copy()
                temp[x] = temp[x]-1
                distance = jsd(p_clusters, np.array(list(temp.values())))
                if distance < best_jsd:
                    best = x
                    best_jsd = distance
            simple_q_cluster_n[best] = simple_q_cluster_n[best]-1
            simple_q_c_total += -1

    return simple_q_cluster_n