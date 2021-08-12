import paddle
import numpy as np
from pgl.utils.logger import log

class History(paddle.nn.Layer):
    r"""History embedding module.
    We locate history embedding in CPU by default, and the rest parts 
    are located in GPU.
    """

    def __init__(self, num_embeddings, embedding_dim, device=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # TODO: device_guard()
        numpy_data = np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32)
        self.emb = paddle.to_tensor(numpy_data, place=paddle.CUDAPinnedPlace())

        self.device = paddle.device.get_device()

        # self.reset_parameters()

    def reset_parameters(self):
        self.emb.set_value(
            np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32))

    @paddle.no_grad()
    def pull(self, n_id):
        out = self.emb
        if n_id is not None:
            assert str(n_id.place)[:4] == str(self.emb.place)[:4]
            out = out.index_select(index=n_id, axis=0)
        return out

    @paddle.no_grad()
    def push(self, x, n_id, offset, count):

        if n_id is None and x.shape[0] != self.num_embeddings:
            raise ValueError

        elif n_id is None and x.shape[0] == self.num_embeddings:
            self.emb.set_value(x)

        elif offset is None or count is None:
            assert str(n_id.place)[:4] == str(self.emb.place)[:4]
            self.emb[n_id] = x.cpu() if str(self.emb.place)[:3] == 'CPU' else x

        else: # Push in chunks
            src_o = 0
            if str(self.emb.place)[:3] == 'CPU':
                x = x.cpu()
            for dst_o, c in zip(offset.tolist(), count.tolist()):
                self.emb[dst_o:dst_o + c] = x[src_o:src_o + c]
                src_o += c
                
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.num_embeddings}, '
                f'{self.embedding_dim}, emb_place={self.emb.place}, '
                f'history_place={self.place})')

