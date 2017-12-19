import tensorflow as tf
from capsule import capsule
from tools import *

def CapsNet(data, batch_size):
    with tf.name_scope("Caps_Net"):
        conv1 = convolution("Conv_1", data, 256, 9, [1,1,1,1])
        caps1 = capsule("Primary_Caps", conv1, batch_size, num_outputs=32, num_units=8, filter_size=9, stride=[1,2,2,1], routing=False, routing_iterations=3)
        caps2 = capsule("DigitCaps", caps1, batch_size, 10, 16, routing=True, routing_iterations=3)
        output = tf.reduce_sum(tf.squeeze(caps2),2)
        #normalize = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis = 2, keep_dims=True))
        #softmax = tf.nnn.softmax(normalize, dim=1)
        #fc = fc_layer("FC_layer", caps2, 10, False)
        return output


