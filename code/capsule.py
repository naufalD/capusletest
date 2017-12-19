import tensorflow as tf
import numpy as numpy
import tools

def capsule(name, data, batch_size, num_outputs, num_units, filter_size = None, stride = None, routing=False, routing_iterations=3):
    with tf.variable_scope(name):
        if not routing:
            #print(data.get_shape())
            capsules = []
            for i in range(num_units):
                with tf.variable_scope("Capsule_%d"%(i)):
                    conv_i = tools.convolution("Caps_Conv", data, num_outputs, filter_size, stride)
                    caps_i = tf.reshape(conv_i, (batch_size,-1,1,1))
                capsules.append(caps_i)
            #print(capsules[0].get_shape())

            capsules = tf.concat(capsules, axis=2)
            #region squash function
            vec_abs = tf.square(capsules)
            vec_abs = tf.reduce_sum(vec_abs)
            vec_abs = tf.sqrt(vec_abs)
            scalar_factor = tf.square(vec_abs)
            scalar_factor = scalar_factor / (1+scalar_factor)

            capsules = scalar_factor*tf.divide(capsules, vec_abs)
            #endregion
            #print(capsules.get_shape())
        else:

            B_IJ = tf.zeros([1, data.get_shape()[1], num_outputs, 1], tf.float32)
            capsules = []
            for j in range(num_outputs):
                #region routing
                W_initializer = tf.random_normal([1, data.get_shape().as_list()[1], data.get_shape().as_list()[2], 16]) # digit caps weight
                W_ij = tf.Variable(W_initializer)
                W_ij = tf.tile(W_ij, [batch_size, 1,1,1])
                u_hat = tf.matmul(W_ij, data, True)

                #print(u_hat.get_shape())

                for r in range(routing_iterations):
                    C_ij = tf.nn.softmax(B_IJ, 2)

                    #print(C_ij.get_shape())

                    B_il, B_ij, B_ir = tf.split(B_IJ, [j, 1, B_IJ.get_shape().as_list()[2]-j-1], axis=2)
                    C_il, C_ij, B_ir = tf.split(C_ij, [j, 1, B_IJ.get_shape().as_list()[2]-j-1], axis =2)

                    #print(C_ij.get_shape())

                    S_j = tf.multiply(C_ij, u_hat)
                    S_j = tf.reduce_sum(S_j, axis = 1, keep_dims=True)

                    #print(S_j.get_shape())

                    #region squash function
                    vec_abs = tf.square(S_j)
                    vec_abs = tf.reduce_sum(vec_abs)
                    vec_abs = tf.sqrt(vec_abs)

                    scalar_factor = tf.square(vec_abs)
                    scalar_factor = scalar_factor / (1+scalar_factor)
                    #endregion

                    #print(S_j.get_shape())

                    V_j = scalar_factor*tf.divide(S_j, vec_abs)

                    V_j_tiled = tf.tile(V_j, [1, data.get_shape().as_list()[1], 1, 1])
                    U_produce_V = tf.matmul(u_hat, V_j_tiled, transpose_a=True) 
                    #print(U_produce_V.get_shape())
                    B_ij += tf.reduce_sum(U_produce_V, axis = 0, keep_dims = True)
                    B_ij = tf.concat([B_il, B_ij, B_ir], axis = 2)
                #endregion
                capsules.append(V_j)

            capsules = tf.concat(capsules, axis=1)

        #print(capsules.get_shape())

        return capsules

def em_routing():
    pass