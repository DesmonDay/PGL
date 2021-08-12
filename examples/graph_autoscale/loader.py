import os
import pdb

import paddle
import numpy as np
from functools import partial

from pgl.utils.logger import log
from pgl.utils.data import Dataset
from pgl.sampling.custom import subgraph
from utils import one_hop_neighbor


class SubgraphData(object):
    def __init__(self, subgraph, batch_size, n_id, offset, count):
        self.subgraph = subgraph
        self.batch_size = batch_size
        self.n_id = n_id
        self.offset = offset
        self.count = count

        
class PartitionDataset(Dataset):
    """Use to return one-batch example"""
    def __init__(self, perm, ptr):
        self.ptr = ptr
        batches_nid = np.split(perm, ptr[1:-1])
        self.batches_nid = [(i, batches_nid[i]) for i in range(len(batches_nid))]

    def __getitem__(self, idx):
        return self.batches_nid[idx]

    def __len__(self):
        return len(self.ptr) - 1

    
def batch_fn(batches_nid, graph, ptr):
    batch_ids, n_ids = zip(*batches_nid)
    batch_ids = np.array(batch_ids)
    n_id = np.concatenate(n_ids, axis=0)
    batch_size = np.size(n_id)
    new_nid, pred_eids = one_hop_neighbor(graph, n_id)
    old_graph_flag = (len(new_nid) == graph.num_nodes) and \
                     (len(pred_eids) == graph.num_edges)
    if not old_graph_flag:
        graph = subgraph(graph, nodes=new_nid, eid=pred_eids)
    offset = ptr[batch_ids]
    count = ptr[batch_ids + 1] - ptr[batch_ids]
    return SubgraphData(graph, batch_size, new_nid, offset, count)


if __name__ == "__main__":
    import paddle
    import pgl
    from utils import random_partition
    from pgl.utils.data.dataloader import Dataloader
    data = pgl.dataset.CoraDataset()
    perm, ptr = random_partition(data.graph, 40)
    dataset = PartitionDataset(perm, ptr)
    collate_fn = partial(batch_fn, graph=data.graph, ptr=ptr)
    loader = Dataloader(dataset, batch_size=10, drop_last=False, shuffle=True, num_workers=1, collate_fn=collate_fn)
    for i, batch_data in enumerate(loader):
        print(i)
        print(batch_data.subgraph)
        print(batch_data.batch_size)
        print(batch_data.n_id)
        print(batch_data.offset)
        print(batch_data.count)
