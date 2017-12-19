import CapsNet
import CifarInput
import tools
import tensorflow as tf
import numpy as np
import os
import threading
from TkinterTest import WindowClass

train_log_dir = r"C:\Projects\Programming\CsDiscount\log"

def run(batch_size =300, learning_rate = 0.01):
    #region create network
    data, label = CifarInput.read_cifar10(r"C:\Projects\Programming\CsDiscount\cifar-10-binary",True, batch_size, True)
    logit = CapsNet.CapsNet(data, batch_size)
    reconstruction = tools.decoder(logit)
    reconstruction_p = tf.placeholder(dtype=tf.float32, shape = [batch_size,32,32,3])
    print("Network Created")
    #endregion

    #region create optimizer
    global_step = tf.Variable(0, trainable=False, name = "global_step")
    loss = tools.loss(logit, label, data, reconstruction_p, batch_size)
    accuracy = tools.accuracy(logit, label)
    train_op = tools.optimize(loss, learning_rate, global_step)
    print("Optimizer Created")
    #endregion

    #region create sessions, queues and savers
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(train_log_dir)
    sess.run(init)
    print("Sessions, Queues and Savers Created")
    #endregion

    
    for x in range (1000):
        print(x)
        reconstruction_run = sess.run(reconstruction)
        sess.run(train_op, feed_dict={reconstruction_p:reconstruction_run})
        if x%5==0:
            mainwindow.newimg(reconstruction_run[0])
            
        if x%100==0:
            print(sess.run(accuracy))
            checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
            saver.save(sess, save_path=checkpoint_path, global_step=x)

#run()

#t = threading.Thread(target = run)
#t.start()
mainwindow = WindowClass()

t = threading.Thread(target=run)
t.start()

mainwindow.start()