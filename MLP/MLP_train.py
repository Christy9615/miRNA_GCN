import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np

# batch iteration function
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    weights = tf.Variable(initial)
    return weights

#bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolutional 2d network
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_pool_2(x, W):
    return tf.nn.max_pool(x, ksize=W, strides=[1, 10, 1, 1], padding="VALID")


# deep neural network
def deepnn(x, keep_prob, args,feature_len):
    # with tf.name_scope('reshape'):
    #     x = tf.reshape(x, [-1, feature_len, 1, 1])
    #
    # with tf.name_scope('conv_pool'):
    #     filter_shape = [4, 1, 1, 4]
    #
    #     W_conv = weight_variable(filter_shape)
    #     b_conv = bias_variable([4])
    #     h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    #     h_pool = tf.nn.max_pool(h_conv, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding="VALID")
    # with tf.device('/cpu:0'):
        '''filter_shape2 = [4,1,4,4]
        W_conv2 = weight_variable(filter_shape2)
        b_conv2 = bias_variable([4])
        h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,4,1,1], strides= [1,4,1,1],padding="VALID")'''

        # regula = tf.contrib.layers.l2_regularizer(args['L2_norm'])
        regula = tf.keras.regularizers.l2(0.0001)
        # h_input1 = tf.reshape(h_pool, [-1, 31 * 4])
        W_fc1 = weight_variable([feature_len, 32])

        b_fc1 = bias_variable([32])
        h_input2 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        h_keep = tf.nn.dropout(h_input2, keep_prob)
        W_fc2 = weight_variable([32, 1])
        b_fc2 = bias_variable([1])
        h_output = tf.matmul(h_keep, W_fc2) + b_fc2
        regularizer = regula(W_fc1) + regula(W_fc2)
        return h_output, regularizer


args = {
    'L2_norm':0.001,
    'learning_rate':0.01, # might be adjust to smaller value
    'batch_size':80000,  # increase may be useful
    'training_epochs':300, # increase to 400-600, 300 has a fast running speed
    'display_step':20,
    'keep_prob':0.9
}


def main(x_train, test_data, y_train):
    with tf.device('/cpu:0'):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        input_data = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        input_label = tf.placeholder(tf.float32, [None, 1])
        keep_prob = tf.placeholder(tf.float32)
        y_conv, losses = deepnn(input_data, keep_prob, args,x_train.shape[1])
        y_res = tf.nn.sigmoid(y_conv)
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=input_label)
        cross_entropy = tf.reduce_mean(cross_entropy)
        los = cross_entropy + losses
        global_step = tf.Variable(100, trainable=False)
        learning_rate = tf.train.exponential_decay(args['learning_rate'], global_step=global_step, decay_steps=10, decay_rate=0.9)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_step = optimizer.minimize(los)
        with tf.name_scope('accuracy'):
            predictions = tf.argmax(y_conv, 1)
            correct_predictions = tf.equal(predictions, tf.argmax(input_label, 1))
            correct_predictions = tf.cast(correct_predictions, tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)

        batch_size = args['batch_size']
        num_epochs = args['training_epochs']
        display_step = args['display_step']
        k_p = args['keep_prob']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

            for i, batch in enumerate(batches):
                x_batch, y_batch = zip(*batch)

                train_step.run(feed_dict={input_data: x_batch, input_label: y_batch, keep_prob: k_p})

                if i % display_step == 0:
                    acc,loss = sess.run([accuracy,los], feed_dict={input_data: x_train, input_label: y_train, keep_prob: k_p})
                    print('after training loss = %f' % loss)
                    print('Accuracy:', acc)

            y_predict = sess.run(y_res, feed_dict={input_data: test_data, keep_prob: 1.0})[:,
                        0]
            tf.get_default_graph().finalize()
            return y_predict

