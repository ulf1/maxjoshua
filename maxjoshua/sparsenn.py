from .fltsel import fltsel
import numpy as np
import itertools
import gc
import tensorflow as tf
from keras_tweaks import dense_sparse_matmul
from typing import List


def pretrain_submodels(X, y,
                       num_out: int = 64,
                       n_select: int = 5,
                       n_draws: int = 100):
    """ Forward selection of N submodels and create params for tf COO tensor

    Example:
    --------
    indices, values, num_in, num_out = pretrain_submodels(
        X, y, num_out=64, n_select=3)
    W = tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=(num_in, num_out))
    """
    # increase draws
    n_draws_ = max(max(X.shape[1] * 2, num_out * 2), n_draws)

    # adjust `y`
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # run mh.fltsel for each target variable
    results = []
    for i in range(y.shape[1]):
        idx, loss, beta, res = fltsel(
            X, y[:, i], preselect=0.8, oob_score=True, subsample=0.5,
            n_select=n_select, unique=True, n_draws=n_draws_)
        print(i, idx, loss, beta)
        results.extend(res)

    # filter unique; select submodel with smallest loss
    ures = {}
    for res in results:
        tmp = ures.get(res[0], {})
        if len(tmp) == 0:
            ures[res[0]] = {"beta": res[1], "loss": res[2]}
        else:
            prevloss = tmp.get("loss", np.inf)
            if prevloss > res[2]:
                ures[res[0]] = {"beta": res[1], "loss": res[2]}

    del results
    gc.collect()

    # convert back
    results = [(r[0], r[1].get('beta'), r[1].get('loss'))
               for r in ures.items()]
    results.sort(key=lambda x: x[-1])
    del ures
    gc.collect()

    # find unused indicies
    allidx = set(range(X.shape[1]))
    selectedidx = set(itertools.chain(*[r[0] for r in results]))
    unused = list(allidx.difference(selectedidx))

    if len(unused) > 0:
        pct_unused = len(unused) / X.shape[1] * 100
        print(f"Unused features: {len(unused)} ({pct_unused:.1f}%)")

    if len(results) < num_out:
        print(f"Insufficent submodels identified: {len(results)} of {num_out}")

    # select the best submodels
    results = results[:num_out]

    # convert to COO indices and values
    indices = []
    values = []
    for j, res in enumerate(results):
        indices.extend([(i, j) for i in res[0]])
        values.extend(res[1])

    # done
    num_in = X.shape[1]
    return indices, values, num_in, num_out


class SparseLayerAsEnsemble(tf.keras.layers.Layer):
    def __init__(self,
                 num_in: int,
                 num_out: int,
                 sp_indices: List[List[int]],
                 sp_values: List[float],
                 sp_trainable: bool = False,
                 **kwargs):
        super(SparseLayerAsEnsemble, self).__init__(**kwargs)
        # layernorm
        self.norm = tf.keras.layers.BatchNormalization(
            center=True, scale=True, trainable=True,
            name="normalize_inputs")
        # sparse tensor
        self.num_in = num_in
        self.num_out = num_out
        self.sp_indices = sp_indices
        self.sp_weights = tf.Variable(
            initial_value=sp_values,
            trainable=sp_trainable,
            name='sparse_weights')

    def _get_sp(self):
        return tf.sparse.SparseTensor(
            dense_shape=(self.num_in, self.num_out),
            indices=self.sp_indices,
            values=self.sp_weights)

    def call(self, inputs: tf.Tensor):
        h = self.norm(inputs)
        W = self._get_sp()
        h = dense_sparse_matmul(h, W)
        return h


class InverseTransformer(tf.keras.layers.Lambda):
    """ Train the inverse transform """
    def __init__(self, units, init_bias=0., init_scale=1.):
        self.units = units
        self.scale = tf.Variable(
            initial_value=tf.ones(self.units) * init_scale,
            trainable=True, name='scale')
        self.bias = tf.Variable(
            initial_value=tf.zeros(self.units) + init_bias,
            trainable=True, name='bias')
        super(InverseTransformer, self).__init__(
            lambda x: x * self.scale + self.bias
        )
