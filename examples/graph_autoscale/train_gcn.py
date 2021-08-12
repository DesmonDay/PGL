import pdb
import time
import tqdm
import argparse
import numpy as np

import pgl
import paddle
from functools import partial
from pgl.utils.logger import log
from paddle.optimizer import Adam

from models import GCN
from loader import PartitionDataset, batch_fn
from utils import random_partition, gen_mask
from pgl.utils.data.dataloader import Dataloader


def load(name, normalized_feature=True):
    if name == 'cora':
        data = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        data = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        data = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(name + " dataset doesn't exists")

    indegree = data.graph.indegree()
    data.graph.node_feat["words"] = normalize(data.graph.node_feat["words"])

    train_mask = gen_mask(data.graph.num_nodes, data.train_index)  # [0 1 1 0 0]
    data.train_mask = paddle.to_tensor(train_mask)
    val_mask = gen_mask(data.graph.num_nodes, data.val_index)
    data.val_mask = paddle.to_tensor(val_mask)
    test_mask = gen_mask(data.graph.num_nodes, data.test_index)
    data.test_mask = paddle.to_tensor(test_mask)
 
    data.label = paddle.to_tensor(np.expand_dims(data.y, -1))
    return data
    

def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def train(dataloader, model, feature, label, train_mask, 
          criterion, optim, epoch, log_per_step=100):
    model.train()

    batch = 0
    total_loss = 0.
    
    for batch_data in dataloader:
        batch += 1

        g = batch_data.subgraph
        batch_size = batch_data.batch_size
        n_id = batch_data.n_id
        offset = batch_data.offset
        count = batch_data.count

        g.tensor()
        n_id = paddle.to_tensor(n_id)
        offset = paddle.to_tensor(offset, place=paddle.CPUPlace())
        count = paddle.to_tensor(count, place=paddle.CPUPlace())
        feat = paddle.gather(feature, n_id)
        pred = model(g, feat, batch_size, n_id, offset, count)
        pred = paddle.gather(pred, n_id[:batch_size])
        train_mask_ = paddle.gather(train_mask, n_id[:batch_size])
        y = paddle.gather(label, n_id[:batch_size])
        """
        # masked_select 目前只支持两个输入相同维度的情况
        loss = criterion(paddle.masked_select(pred, train_mask),
                         paddle.masked_select(label, train_mask))
        """
        true_index = np.argwhere(train_mask_.numpy()) 
        if true_index.shape[0] == 0:
            continue
        true_index = paddle.to_tensor(true_index)
        pred = paddle.gather(pred, true_index)
        y = paddle.gather(y, true_index)
        loss = criterion(pred, y) 
        loss.backward()
        optim.step()
        optim.clear_grad()
        log.info("Epoch %d Batch %s %s" % (epoch, batch, loss.numpy()))


@paddle.no_grad()
def test():
    model.eval()

    # Full-batch inference since the graph is small
    out = model()


def main(args):  
    # Data Process
    pdb.set_trace()
    data = load(args.dataset, args.feature_pre_normalize)
    feature = data.graph.node_feat["words"]
    perm, ptr = random_partition(data.graph, num_clusters=1, shuffle=True)
    dataset = PartitionDataset(perm, ptr)
    collate_fn = partial(batch_fn, graph=data.graph, ptr=ptr)
    train_loader = Dataloader(dataset, 
                              batch_size=10, 
                              drop_last=False, 
                              shuffle=True, 
                              num_workers=1, 
                              collate_fn=collate_fn)
    graph = data.graph
    feature = paddle.to_tensor(feature)

    # Use GCN+GAS model
    model = GCN(
        num_nodes=graph.num_nodes,
        in_channels=graph.node_feat["words"].shape[1],
        hidden_channels=16,
        out_channels=data.num_classes,
        num_layers=1,
        dropout=0.5,
        pool_size=2,
        buffer_size=2000,
    )    

    # Define optimizer and loss
    optim = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=args.weight_decay)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    
    for epoch in range(args.epoch):
        train(train_loader, model, feature, data.label, data.train_mask,
              criterion, optim, epoch)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Testing GAS')
    parser.add_argument(
        "--dataset", type=str, default="cora")
    parser.add_argument(
        "--epoch", type=int, default=100, help="Epoch")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay rate")
    parser.add_argument(
        "--feature_pre_normalize",
        type=bool,
        default=True,
        help="pre_normaliza feature")
    args = parser.parse_args()
    log.info(args)
    main(args)

