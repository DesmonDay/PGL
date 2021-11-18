# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021, rusty1s(github).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Dataset and DataLoader for GNNAutoScale.
"""

import os
from functools import partial

import numpy as np
import paddle
import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataset
from pgl.sampling.custom import subgraph
from pgl.utils.data.dataloader import Dataloader
from pgl.utils.transform import to_undirected, add_self_loops

from utils import generate_mask


class SubgraphData(object):
    """Encapsulating the basic subgraph data structure.

    Args:

        subgraph (pgl.Graph): Subgraph of the original graph object.

        batch_size (int): Original number of nodes of batch partition graphs.

        n_id (numpy.ndarray): An 1-D numpy array, contains original node ids of batch partition graphs, and 
                            the corresponding one-hop neighbor node ids of the original nodes.

        offset (numpy.ndarray): An 1-D numpy array, means the begin points of parition graphs' nodes in history embeddings.

        count (numpy.ndarray): An 1-D numpy array, contains the number of nodes of parition graphs in history embeddings.

    Examples:

        - Suppose after graph partition and permutation, new node order is [4, 6, 1, 5, 7, 0, 3, 2, 8, 9].
          And the partition graphs are [4, 6], [1, 5, 7], [0, 3], [2, 8, 9].

        - Suppose we have a batch subgraph, and its nodes is [0, 3, 4, 6](two of the above partition graphs), then the batch_size = 4.

        - We find the one hop neighbor nodes of the subgraph's nodes, which are [1, 2, 4, 7, 9], 
          then the n_id is [0, 3, 4, 6, 1, 2, 7, 9]. Also `4` is also the neighbor nodes, it is included in batch_size. 

        - The offset of the batch subgraph is [5, 1], we can see they are position indexes of 0 and 4 in node order.

        - The count of the batch subgraph is [2, 2], which is the length of corresponding partition graphs.

    """

    def __init__(self, subgraph, batch_size, n_id, offset, count):
        self.subgraph = subgraph
        self.batch_size = batch_size
        self.n_id = n_id
        self.offset = offset
        self.count = count


class PartitionDataset(Dataset):
    """PartitionDataset helps build train dataset.

    Args:

       part (numpy.ndarray): An 1-D numpy array, which helps distinguish different parts of partition graphs.
    
    """

    def __init__(self, part):
        self.part = part

    def __getitem__(self, idx):
        return (idx, np.arange(self.part[idx], self.part[idx + 1]))

    def __len__(self):
        return len(self.part) - 1


class EvalPartitionDataset(Dataset):
    """EvalPartitionDataset helps build eval and test dataset, 
       which can be generated in advance for better inference speed.

    Args:

       graph (pgl.Graph): Evaluation subgraph.
  
       part (numpy.ndarray): An 1-D numpy array, which helps distinguish different parts of partition graphs. 

       batch_size (int): Eval batch size, usually the same with train batch size.

       node_buffer (numpy.ndarray): An intermediate node buffer mainly used for finding out-of-batch neighbors of batch graphs.

    """

    def __init__(self, graph, part, batch_size, node_buffer):
        self.part = part[::batch_size]

        num_nodes = graph.num_nodes
        if self.part[-1] != num_nodes:
            self.part = paddle.concat([
                self.part, paddle.to_tensor(
                    [num_nodes], dtype=self.part.dtype)
            ])
        batches_nid = np.split(np.arange(num_nodes), self.part[1:-1])
        batches_nid = [(i, batches_nid[i]) for i in range(len(batches_nid))]

        collate_fn = partial(
            subdata_batch_fn,
            graph=graph,
            part=self.part,
            node_buffer=node_buffer)
        self.dataloader_list = list(
            Dataloader(
                dataset=batches_nid,
                batch_size=1,
                num_workers=5,
                shuffle=False,
                collate_fn=collate_fn))

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.part) - 1


def one_hop_neighbor(graph, nodes, node_buffer):
    """Find one hop neighbors.
    """

    pred_nodes, pred_eids = graph.predecessor(nodes, return_eids=True)
    pred_nodes = np.concatenate(pred_nodes, -1)
    pred_eids = np.concatenate(pred_eids, -1)

    node_buffer[nodes] = 1
    out_of_batch_neighbors = pred_nodes[node_buffer[pred_nodes] == 0]
    out_of_batch_neighbors = np.unique(out_of_batch_neighbors)
    new_nodes = np.concatenate((nodes, out_of_batch_neighbors))
    node_buffer[nodes] = 0
    return new_nodes, pred_eids


def subdata_batch_fn(batches_nid, graph, part, node_buffer):
    """Basic function for creating batch subgraph data.
    """

    batch_ids, n_ids = zip(*batches_nid)
    batch_ids = np.array(batch_ids)
    n_id = np.concatenate(n_ids, axis=0)
    batch_size = np.size(n_id)
    new_nid, pred_eids = one_hop_neighbor(graph, n_id, node_buffer)
    sub_graph = subgraph(graph, nodes=new_nid, eid=pred_eids)
    offset = part[batch_ids]
    count = part[batch_ids + 1] - part[batch_ids]
    return SubgraphData(sub_graph, batch_size, new_nid, offset, count)


def load_dataset(data_name):
    """Load dataset.

    Args:

        data_name (str): The name of dataset.

    Returns:

        dataset (pgl.dataset): Return the corresponding dataset, containing graph information, feature, etc.

        mode (str): Currently we have 's' and 'm' mode, which mean small dataset and medium dataset respectively. 
        
    """

    data_name = data_name.lower()
    mode = None
    if data_name == 'reddit':
        mode = 'm'
        dataset = pgl.dataset.RedditDataset()
        y = np.zeros(dataset.graph.num_nodes, dtype="int64")
        y[dataset.train_index] = dataset.train_label
        y[dataset.val_index] = dataset.val_label
        y[dataset.test_index] = dataset.test_label
        dataset.y = y
    elif data_name == 'arxiv':
        mode = 'm'
        dataset = pgl.dataset.OgbnArxivDataset()
        dataset.graph = to_undirected(dataset.graph, copy_node_feat=False)
        dataset.graph = add_self_loops(dataset.graph, copy_node_feat=False)
    elif data_name == 'cora':
        mode = 's'
        dataset = pgl.dataset.CoraDataset()
    elif data_name == 'pubmed':
        mode = 's'
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif data_name == 'citeseer':
        mode = 's'
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(data_name + " dataset doesn't exist currently.")

    if mode == 's':

        def normalize(feat):
            return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)

        indegree = dataset.graph.indegree()
        dataset.graph.node_feat["words"] = normalize(dataset.graph.node_feat[
            "words"])
        dataset.feature = dataset.graph.node_feat["words"]

    dataset.train_mask = generate_mask(dataset.graph.num_nodes,
                                       dataset.train_index)
    dataset.val_mask = generate_mask(dataset.graph.num_nodes,
                                     dataset.val_index)
    dataset.test_mask = generate_mask(dataset.graph.num_nodes,
                                      dataset.test_index)

    return dataset, mode


def create_dataloaders(graph, mode, part, num_workers, config):
    """Create train loader and eval loader for different datasets.
 
    **Notes**:
        For extremely large dataset like ogbn-papers100m, we will add a new example in near future,
        since current eval_loader might not be suitable for large dataset.

    """

    train_dataset = PartitionDataset(part)
    collate_fn = partial(
        subdata_batch_fn,
        graph=graph,
        part=part,
        node_buffer=np.zeros(
            graph.num_nodes, dtype="int64"))
    train_loader = Dataloader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn)
    if config.gen_train_data_in_advance:
        # For relatively small dataset, we can generate train data in advance.
        train_loader = list(train_loader)

    eval_dataset = EvalPartitionDataset(
        graph,
        part,
        config.batch_size,
        node_buffer=np.zeros(
            graph.num_nodes, dtype="int64"))
    eval_loader = eval_dataset.dataloader_list

    return train_loader, eval_loader
