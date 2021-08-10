import paddle
import math
import numpy as np


def random_partition(graph, num_clusters, shuffle=True):
    r"""Randomly partition graph into small clusters, returning its
    "clustered" permutation `perm` and corresponding cluster slices
    `ptr`."""
    # TODO: 是否应该返回    
    num_nodes = graph.num_nodes

    if num_clusters <= 1:
        perm, ptr = np.arange(num_nodes), np.array([0, num_nodes])
    else:
        perm = np.arange(0, num_nodes)
        if shuffle:
            np.random.shuffle(perm)
        # get ptr
        cs = int(math.ceil(num_nodes / num_clusters))
        ptr = [cs * i if cs * i <= num_nodes else num_nodes 
               for i in range(num_clusters + 1)]
        ptr = np.array(ptr)

    return perm, ptr


def one_hop_neighbor(graph, n_id):
    r"""Get one hop neighbors for n_id, we get new `n_id` tensor: 
    [in-batch n_id | out_batch n_id]"""
    pred_nodes, pred_eids = graph.predecessor(n_id, return_eids=True)
    all_pred_nodes = []
    for pred in pred_nodes:
        all_pred_nodes.extend(pred)
    all_pred_nodes = set(all_pred_nodes)
    all_pred_eids = []
    for eids in pred_eids:
        all_pred_eids.extend(eids)
    all_pred_eids = list(set(all_pred_eids))
    out_of_batch_neighbors = []
    for p in all_pred_nodes:
        if p not in n_id:
            out_of_batch_neighbors.append(p)
    out_of_batch_neighbors = np.array(out_of_batch_neighbors)
    new_n_id = np.concatenate((n_id, out_of_batch_neighbors))
    all_pred_eids = np.array(all_pred_eids)
    return new_n_id, all_pred_eids


def gen_mask(num_nodes, index):
    mask = np.zeros(num_nodes, dtype=np.int32)
    mask[index] = 1
    return mask
