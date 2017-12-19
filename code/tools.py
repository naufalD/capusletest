import tensorflow as tf
import numpy as np

def convolution(name, data, filter_number, filter_size, stride):
    with tf.variable_scope(name):
        filters = tf.get_variable(
        name="filters", 
        shape=[filter_size, filter_size, data.get_shape()[-1], filter_number],
        dtype= tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(), 
        trainable=True)
        bias = tf.get_variable(name="bias",
                               shape=[filter_number], 
                               dtype= tf.float32, 
                               initializer=tf.constant_initializer(0), 
                               trainable=True)

        data = tf.nn.conv2d(data, filters, stride, padding = "SAME", name="Convolution")
        data = tf.nn.bias_add(data, bias, name="Bias_add")
        data = tf.nn.relu(data, name="Relu")
    return data

def fc_layer(layer_name, data, output_size, dropout, activation=None):
    with tf.variable_scope(layer_name):
        if (len(data.get_shape())>2):
            data = tf.reshape(data, [data.get_shape()[0].value, -1])

        weights = tf.get_variable(name="weight", 
                                  shape=[data.get_shape()[1], output_size], 
                                  dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  trainable=True)
        bias = tf.get_variable(name="bias", 
                               shape=[output_size], 
                               dtype=tf.float32, 
                               initializer=tf.constant_initializer(0), 
                               trainable=True)
        #weights = tf.nn.dropout(weights, dropout)
        data = tf.matmul(data, weights)
        data = tf.nn.bias_add(data, bias)

        if (activation == 'relu'):
            data = tf.nn.relu(data, name='Relu')

        return data

def loss(logits, label, images, reconstruction, batch_size):
    with tf.name_scope("Loss"):
        #data = tf.nn.softmax_cross_entropy_with_logits(logits=data, labels=label)
        #data = tf.reduce_mean(data, name="Loss_Function")
        #tf.summary.scalar("Loss", data)
        left = tf.nn.relu(0.9 - logits) ** 2
        right = tf.nn.relu(logits - 0.1) ** 2
        print(left.shape, right.shape, label.shape)

        margin_loss = label * left + 0.5 * (1.0 - label) * right
        margin_loss = tf.reduce_sum(margin_loss)

        reconstruction_loss = tf.losses.mean_squared_error(images, reconstruction)

    return (margin_loss + 0.0005 * reconstruction_loss) / batch_size

def accuracy(data, label):
    with tf.name_scope("Accuracy"):
        data = tf.equal(tf.argmax(data, 1), tf.argmax(label, 1))
        data = tf.cast(data, tf.float32)
        data = tf.reduce_mean(data)*100
        tf.summary.scalar("Accuracy", data)

        return data

def optimize(loss, learning_rate, global_step):
    with tf.name_scope("Optimizer"):
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

def decoder(logits):
    with tf.variable_scope("decoder"):
        data = fc_layer("Layer_1", logits, 512, 0, "relu")
        data = fc_layer("Layer_2", data, 1024, 0, "relu")
        data = fc_layer("Layer_3", data, 3072, 0)
        data = tf.nn.sigmoid(data)
        data = tf.reshape(data, [data.get_shape()[0], 32,32,3])
        
        return data