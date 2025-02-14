import numpy as np
import math
import time
import logging
import torch

class Matrix:

    def variables(self):
        v = []
        v.extend(self._shardIndices)
        v.extend(self._shardValues)
        v.append(self._nnz)
        v.append(self._shardCount)
        return v

    def initializers(self):
        if self.is_chief:
            return [self._shardCount.initializer, self._nnz.initializer]
        else:
            return []

    def matmul(self, dense, transpose_b = False):

        if self._is_dense:
            if transpose_b == True:
                r = torch.mm(self._shardSparseTensor[0], dense.t())
            else:
                r = torch.mm(self._shardSparseTensor[0], dense)
        else:

            if not transpose_b:
                b = dense
            else:
                b = dense.t()

            # return torch.randn(self.dense_shape[0], b.shape[1], device='cuda')
            r = torch.mm(self._shardSparseTensor[0], b)

            for i in range(1, len(self._shardSparseTensor)):
                r = r + torch.mm(self._shardSparseTensor[i], b)

        return r

    def _initializeVariables(self, name, dense_shape, cach_size, device, coo_matrix = None):

        indices = None
        values = None
        if not coo_matrix is None:
            indices = np.vstack((coo_matrix.row,coo_matrix.col))
            values = coo_matrix.data

        n_nnz = int(values.shape[-1])
        n_shard = int(math.ceil(n_nnz / cach_size) if cach_size > 0 else 1)

        init = np.empty((1), dtype=np.int32)

        init[0] = n_shard
        self._shardCount = torch.from_numpy(init).to(device)
        init[0] = n_nnz
        self._nnz = torch.from_numpy(init).to(device)

        # sess.run([self._shardCount.initializer, self._nnz.initializer])



        if n_shard == 1 and n_nnz >= 0.5 * dense_shape[0] * dense_shape[1]:

            init_matrix = coo_matrix.todense()

            spa = torch.from_numpy(init_matrix).to(device)

            self._shardValues.append(spa)
            self._shardSparseTensor.append(spa)
            self._is_dense = True

            logging.debug('Convert sparse tensor ' + str(name) + ' to dense.')

        else:
            for n in range(n_shard):

                length = min(n_nnz, (n + 1) * self._cach_size) - n * self._cach_size if self._cach_size > 0 else n_nnz

                idx_init = indices[:,n * self._cach_size : n * self._cach_size + length]
                val_init = values[n * self._cach_size : n * self._cach_size + length]

                idx = torch.LongTensor(idx_init).to(device)
                val = torch.FloatTensor(val_init).to(device)

                # spa = torch.sparse.FloatTensor(idx, val, dense_shape)
                spa = torch.sparse.FloatTensor(idx, val, dense_shape).coalesce()


                # Warning : will seriously slow down the speed
                # spa = tf.sparse.reorder(spa)

                self._shardIndices.append(idx)
                self._shardValues.append(val)
                self._shardSparseTensor.append(spa)

    def __init__(self, name, dense_shape, cach_size, device, coo_matrix=None):


        self._shardIndices = []
        self._shardValues = []
        self._shardSparseTensor = []

        self.name = name
        self.dense_shape = dense_shape
        self.shape = self.dense_shape
        self._cach_size = cach_size

        self._is_dense = False


        self._initializeVariables(name, dense_shape, cach_size, device, coo_matrix)










