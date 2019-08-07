#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:11:37 2019

@author: TH
"""

# Model restore

import tensorflow as tf

var1 = tf.Variable([0,0], name = "v1")
var2 = tf.Variable([0,0], name = "v2")

saver = tf.train.Saver()
module_file = tf.train.latest_checkpoint("test/")

with tf.Session() as sess:
    saver.restore(sess, module_file)
    print ("Model Restored.")
    
#%%
    
a = tf.placeholder(tf.float32, shape = [2], name = None)
b = tf.constant([6, 4], tf.float32)

c = tf.add(a,b)

with tf.Session() as sess:
    print (sess.run(c, feed_dict = {a:[10, 10]}))
    
#%%
    
a = tf.constant(5)
b = tf.constant(6)
c = tf.constant(4)

add = tf.add(b, c)
mul = tf.multiply(a, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print (result)