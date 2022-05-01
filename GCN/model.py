import tensorflow.compat.v1 as tf
from GCN.layer import GraphConvolutionSparse, GraphConvolution

flags = tf.flags
FLAGS = flags.FLAGS

# Model class
# One hidden layer, one embedding layer, one decoder

class GCNModel():
    def __init__(self, placeholders, num_features, features_nonzero, name):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.layer1=None
        self.loss=0
        self.layer2=None
        with tf.variable_scope(self.name):
            self.build()

    def build(self):

        # Input dim : num_feature
        # Output dim : FLAGS.hidden1
        # activation function : relu
        
        self.layer1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout)
        self.embeddings = self.layer1((self.inputs))
        
        # Leave the module for possible second layer
        # activate funtion : return x directly

        # self.layer2 = GraphConvolution(
        #     name='gcn_dense_layer',
        #     input_dim=FLAGS.hidden1,
        #     output_dim=FLAGS.hidden2,
        #     adj=self.adj,
        #     act=lambda x: x, 
        #     dropout=self.dropout)
        # self.embeddings=self.layer2(self.hidden1)

        # Decoder
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden1,
            act=lambda x: x)(self.embeddings)  # call function will be used here

        for var in self.layer1.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Leave for second layer in the future (If we have one)   
        # for var in self.layer2.vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)



# Decoder class

class InnerProductDecoder():

    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)  # Use dropout layer to prevent overfitting
            x = tf.transpose(inputs)
            x = tf.matmul(inputs, x)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs
