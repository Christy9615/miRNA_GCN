import tensorflow.compat.v1 as tf
flags = tf.flags
FLAGS = flags.FLAGS

# This is Optimizer

class Optimizer():
    def __init__(self, preds, labels, num_nodes, num_edges,l2_loss,mask):
        # pos_weight = float(num_nodes ** 2 - num_edges) / num_edges
        # norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)

        # Use tf.reduce_mean to calculate average value
        self.cost =tf.nn.sigmoid_cross_entropy_with_logits(
                logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        self.cost *= mask
        self.cost=l2_loss+tf.reduce_mean(self.cost)

        # This is another way to get the cost with weighted corss entropy, leave it here
        # in case current one not working

        # self.cost = l2_loss+norm * tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(
        #         logits=preds, targets=labels, pos_weight=pos_weight))

        # Use Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        # Minimize cost
        self.opt_op = self.optimizer.minimize(self.cost)

        # Calculate gradient
        self.grads_vars = self.optimizer.compute_gradients(self.cost)