import sys
import pdb
import warnings

import paddle
from pgl.utils.logger import log

sys.path.append("..") 
from history import History
from pool import AsyncIOPool


class ScalableGNN(paddle.nn.Layer):
    def __init__(self, num_nodes, hidden_channels, num_layers, 
                 pool_size=None, buffer_size=None, device=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size

        self.histories = paddle.nn.LayerList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers - 1)
        ])

        self.pool = None
        self._async = False
        self.__out = None

        self._init_pool()

    @property
    def emb_device(self):
        return self.histories[0].emb.place

    @property
    def device(self):
        return self.histories[0].device

    def _init_pool(self):
        # Only initialize the AsyncIOPool in case histories are on CPU.
        if (str(self.emb_device) == 'CUDAPinnedPlace' 
                and str(self.device[:3]) == 'gpu'
                and self.pool_size is not None 
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim)

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()

    def __call__(self, subgraph, x, batch_size=None, n_id=None, offset=None, 
                 count=None, loader=None, **kwargs):
        if loader is not None:
            return self.mini_inference(loader)
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        if (batch_size is not None and not self._async
                and str(self.emb_device)[:3] == 'CUDAPinnedPlace'
                and str(self.device)[:3] == 'gpu'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        out = None
     
        # 1. 满足 async 条件，进行异步 pull
        
        if self._async:
            for hist in self.histories:
                x_id = n_id[batch_size:].pin_memory()
                self.pool.async_pull(hist.emb, x_id, None, None)
        
        # 2. 进入各自 model 的 forward 函数，进行 push_and_pull
        out = self.forward(subgraph, x, batch_size, n_id, offset, count, **kwargs)
        """
        # 3. 同样满足 async 条件，最后进行同步push
        if self._async:
            for hist in self.histories:
                self.pool.synchronize_push()
        """
        self._async = False
        return out

    def push_and_pull(self, history, x, batch_size=None, n_id=None, 
                      offset=None, count=None):
        if n_id is None and x.shape[0] != self.num_nodes:
            return x

        if n_id is None and x.shape[0] == self.num_nodes:
            history.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
            return paddle.concat([x[:batch_size], h], axis=0)

        else:
            out_batch_size = int(n_id.numel().numpy()[0] - batch_size)
            out = self.pool.synchronize_pull()[:out_batch_size]
            self.pool.async_push(x[:batch_size], history.emb, offset, count)
            out = paddle.concat([x[:batch_size], out], axis=0)
            self.pool.free_pull()
            return out

    @property
    def _out(self):
        if self.__out is None:
            self.__out = paddle.empty(shape=[self.num_nodes, self.out_channels])
            self.__out = self.__out.pin_memory()
        return self.__out

    @paddle.no_grad()
    def mini_inference(self, feature, loader):
        r"""An implementaion of layer-wise evaluation of GNNs."""
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.
        loader = [sub_data + ({}, ) for sub_data in loader]

        # We push the outputs of the first layer to the history
        for batch_data, state in loader:
            out = self.forward_layer(0, feature, batch_data.subgraph, state)[:batch_data.batch_size]
            self.pool.async_push(out, self.histories[0].emb, batch_data.offset, batch_data.count)
        self.pool.synchronize_push()

        for i in range(1, len(self.histories)):
            # Pull the complete layer-wise history:
            for batch_data, state in loader:
                self.pool.async_pull(self.histories[i - 1].emb, batch_data.n_id[batch_data.batch_size:],
                                     batch_data.offset, batch_data.count)
            # Compute new output embeddings one-by-one and start pushing them     
            # to the history.
            for batch_data, state in loader:
                x = self.pool.synchronize_pull()[:batch_data.n_id.numel().numpy()[0]]
                out = self.forward_layer(i, x, batch_data.subgraph, state)[:batch_data.batch_size]
                self.pool.async_push(out, self.histories[i].emb, offset, count)
                self.pool.free_pull()
            self.pool.synchronize_push()

        # We pull the histories from the last layer:
        for batch_data, state in loader:
            self.pool.async_pull(self.histories[-1].emb, batch_data.n_id[batch_data.batch_size:],
                                 batch_data.offset, batch_data.count)

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for batch_data, state in loader:
            x = self.pool.synchronize_pull()[:batch_data.n_id.numel().numpy()[0]]
            out = self.forward_layer(self.num_layers - 1, x, batch_data.subgraph,
                                     state)[batch_data.batch_size]
            self.pool.async_push(out, self._out, offset, count)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out

    @paddle.no_grad()
    def forward_layer(self, layer, x, subgraph, state):
        raise NotImplementedError
