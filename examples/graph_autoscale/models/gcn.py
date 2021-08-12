import pdb
import pgl
import paddle
import paddle.nn as nn

from .base import ScalableGNN

class GCN(ScalableGNN):
    def __init__(self, num_nodes, in_channels, hidden_channels, 
                 out_channels, num_layers, dropout=0.0, batch_norm=False,
                 pool_size=None, buffer_size=None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                        buffer_size, device)
       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
       
        self.gcns = nn.LayerList()
        self.drops = nn.LayerList()
        for i in range(num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        in_channels,
                        hidden_channels,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        hidden_channels,
                        hidden_channels,
                        activation="relu",
                        norm=True))
            self.drops.append(nn.Dropout(self.dropout))
        self.gcns.append(pgl.nn.GCNConv(hidden_channels,
                                        out_channels))

        self.bns = nn.LayerList()
        for i in range(num_layers):
            bn = nn.BatchNorm1D(hidden_channels)
            self.bns.append(bn)

    def forward(self, graph, x, *args):
        for gcn, bn, drop, hist in zip(self.gcns[:-1], self.bns, self.drops, self.histories):
            x = gcn(graph, x)
            if self.batch_norm:
                x = bn(x)
            x = self.push_and_pull(hist, x, *args)
            x = drop(x)

        h = self.gcns[-1](graph, x)

        return h
      
    @paddle.no_grad()
    def forward_layer(self, layer, graph, x, state):
        """
        layer: gnn layer index
        """
        h = self.gcns[layer](graph, x)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            h = self.drops[layer](h)
        else:
            h = self.gcns[layer](graph, x)
        
        return h
