import pdb

import paddle
from paddle.device import cuda
from pgl.utils.logger import log

async_read = paddle.fluid.framework.core.async_read
async_write = paddle.fluid.framework.core.async_write


class AsyncIOPool(paddle.nn.Layer):
    def __init__(self, pool_size, buffer_size, embedding_dim):
        super().__init__()

        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.embedding_dim = embedding_dim

        self._pull_queue = []
        self._push_cache = [None] * pool_size
        self._push_streams = [None] * pool_size
        self._pull_streams = [None] * pool_size
        self._cpu_buffers = [None] * pool_size
        self._cuda_buffers = [None] * pool_size
        self._pull_index = -1
        self._push_index = -1

    def _pull_stream(self, idx):
        if self._pull_streams[idx] is None:
            self._pull_streams[idx] = cuda.Stream(flag=1)
            log.info("New Pull Stream")
        return self._pull_streams[idx]

    def _push_stream(self, idx):
        if self._push_streams[idx] is None:
            self._push_streams[idx] = cuda.Stream(flag=1)
            log.info("New Push Stream")
        return self._push_streams[idx]

    def _cpu_buffer(self, idx):
        if self._cpu_buffers[idx] is None:
            self._cpu_buffers[idx] = paddle.empty(
                                        shape=[self.buffer_size, self.embedding_dim])
            self._cpu_buffers[idx] = self._cpu_buffers[idx].pin_memory()
        return self._cpu_buffers[idx]

    def _cuda_buffer(self, idx):
        if self._cuda_buffers[idx] is None:
            self._cuda_buffers[idx] = paddle.empty(
                                        shape=[self.buffer_size, self.embedding_dim])
        return self._cuda_buffers[idx]

    @paddle.no_grad()
    def async_pull(self, src, index, offset, count):
        self._pull_index = (self._pull_index + 1) % self.pool_size
        data = (self._pull_index, src, index, offset, count)
        self._pull_queue.append(data)
        if len(self._pull_queue) <= self.pool_size:
            self._async_pull(self._pull_index, src, index)

    @paddle.no_grad()
    def _async_pull(self, idx, src, index):
        with cuda.stream_guard(self._pull_stream(idx)):
            async_read(src, self._cuda_buffer(idx), index, self._cpu_buffer(idx))

    @paddle.no_grad()
    def synchronize_pull(self):
        # TODO: doubt 
        idx = self._pull_queue[0][0]
        cuda.synchronize() 
        self._pull_stream(idx).synchronize()
        return self._cuda_buffer(idx)

    @paddle.no_grad()
    def free_pull(self):
        # Free the buffer space and start pulling from remaining queue
        self._pull_queue.pop(0)
        if len(self._pull_queue) >= self.pool_size:
            data = self._pull_queue[self.pool_size - 1]
            idx, src, index, offset, count = data
            self._async_pull(idx, src, index)
        elif len(self._pull_queue) == 0:
            self._pull_index = -1

    @paddle.no_grad()
    def async_push(self, src, dst, offset, count):
        self._push_index = (self._push_index + 1) % self.pool_size
        self.synchronize_push(self._push_index)
        self._push_cache[self._push_index] = src
        with cuda.stream_guard(self._push_stream(self._push_index)):
            # GPU写CPU
            async_write(src, dst, offset, count)

    @paddle.no_grad()
    def synchronize_push(self, idx):
        if idx is None:
            for idx in range(self.pool_size):
                self.synchronize_push(idx)
            self._push_index = -1
        else:
            self._push_stream(idx).synchronize()
            self._push_cache[idx] = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return (f'{self.__class__.__name__}(pool_size={self.pool_size}, '
                f'buffer_size={self.buffer_size}, '
                f'embedding_dim={self.embedding_dim}, ')
