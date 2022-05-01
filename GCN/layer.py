import tensorflow.compat.v1 as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS


class GraphConvolutionSparse():
    # Graph convolution layer for sparse inputs

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')  # Generate weight
        self.dropout = dropout
        self.adj = adj
        self.act = act  # activation function
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)  # dropput some feature
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])  # xw
            x = tf.sparse_tensor_dense_matmul(self.adj, x)  # Z=D-/2AD1/2XW
            outputs = self.act(x)  # activation function
        return outputs


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')  # init the weight
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            # multiply the matrix
            x = tf.sparse_tensor_dense_matmul(self.adj, x)

            outputs = self.act(x)
        return outputs


# Decoder for inner product


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, input_dim, name='weights')  # init the weight

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            x = tf.matmul(inputs, self.vars['weights'])
            x = tf.matmul(x, tf.transpose(inputs))
            x = tf.reshape(x, [-1])  # reshape to 1d matrix
            outputs = self.act(x)
        return outputs

# Init the weight

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)  # output random value
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)  # use tf.floor return biggest number <= x
    pre_out = tf.sparse_retain(x, dropout_mask)  # keep a non-empty value in SparseTensor
    return pre_out * (1. / keep_prob)
