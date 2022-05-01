from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from GCN.model import GCNModel
from GCN.optimizer import Optimizer

# Settings

flags = tf.flags
tf.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')  # conv layer1's output_dim, conv layer2's input_dim
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')  # use dropput to prevent overfitting
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
tf.compat.v1.disable_eager_execution()

# Build mask matrix

def construct_mask(adj, ratio):
    if ratio == 0:
        return adj
    none_zero_position = np.where(adj != 0)
    zero_position = np.where(adj == 0)
    zero_position_row = zero_position[0]
    zero_position_col = zero_position[1]
    negative_randomlist = np.random.permutation(zero_position_row.shape[0])
    negative_index = negative_randomlist[0:ratio * len(none_zero_position[0])]
    adj[zero_position_row[negative_index], zero_position_col[negative_index]] = 1
    return adj

# Convert sparse matrix to tuple
# https://blog.csdn.net/csdn15698845876/article/details/73380803

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row,
                        sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Preprocess the graph

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

# Build dict

def construct_feed_dict(adj_normalized, adj, features, adj_mask, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    feed_dict.update({placeholders['mask']: adj_mask})
    return feed_dict


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))



# Build a simple GCN module

# reference: 
# http://snap.stanford.edu/deepnetbio-ismb/


def get_new_scoring_matrices(adj, feature_mat=None):
    emb = get_gcn_emb(adj, feature_mat)
    return sigmoid(np.dot(emb, emb.T))


def get_gcn_emb(adj, adj_mask=None,feature_mat=None):
    tf.reset_default_graph()
    if adj_mask is None:
        adj_mask = construct_mask(np.copy(adj), 0)  # mask matrix
    else:
        adj_mask=construct_mask(np.copy(adj_mask),0)
    adj_mask = sp.csr_matrix(adj_mask)
    adj = sp.csr_matrix(adj)  # convert to sparse matrix
    num_nodes = adj.shape[0]  # get num of nodes
    num_edges = adj.sum()  # get num of edges
    # Featureless
    if feature_mat is None:
        features = sparse_to_tuple(sp.identity(num_nodes))
    else:
        features = sparse_to_tuple(sp.csr_matrix(feature_mat))

    num_features = features[2][1]  # get num of features
    features_nonzero = features[1].shape[0]
    adj_orig = adj - sp.dia_matrix((adj.diagonal(), [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()  # remove 0 elements in matrix
    adj_norm = preprocess_graph(adj)  # normlize D-1/2AD-/2

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'mask': tf.sparse_placeholder(tf.float32)
    }

    # init GCN model
    model = GCNModel(placeholders, num_features, features_nonzero, name='yeast_gcn')

    # init optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            num_nodes=num_nodes,
            num_edges=num_edges,
            l2_loss=model.loss,
            mask=tf.reshape(tf.sparse_tensor_to_dense(placeholders['mask'], validate_indices=False), [-1])
        )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # init all global veriables
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_mask = sparse_to_tuple(adj_mask)

    # Train model
    for epoch in range(FLAGS.epochs):
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, adj_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch%50==0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost)
                  )

    print('Optimization Finished!')
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    sess.close()
    tf.get_default_graph().finalize()
    return emb
