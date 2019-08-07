#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:40:09 2019

@author: TH
"""

import tensorflow as tf

m1 = tf.constant([3,5])
m2 = tf.constant([2,4])

result = tf.add(m1, m2)
print (result)

#%%

sess = tf.Session()
print (sess.run(result))

sess.close()

#%%

# Additional way of comuting 

with tf.Session() as sess:
    res = sess.run([result])

print (res)

#%%

# Assign specific GPU to compute

# =============================================================================
# =============================================================================
with tf.Session() as sess:
    with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
        m1 = tf.constant([3,5])
        m2 = tf.constant([2,4])
        result = tf.add(m1,m2)
# =============================================================================
# =============================================================================
print (result)

#%%
        
# as.default  generate default 
m1 = tf.constant([3,5])
m2 = tf.constant([2,4])

result = tf.add(m1, m2)   
sess1 = tf.Session()
with sess1.as_default():
    print (result.eval())
        
#%%
    
sess = tf.InteractiveSession()
print (result.eval())

#%%

# =============================================================================
# Tensor
# =============================================================================

a = tf.constant([[2.0, 3.0]], name = "a")
b = tf.constant([[1.0], [4.0]], name = "b")

result = tf.matmul(a,b,name = "mul")

print (result)

#%%

a = tf.constant([2.0, 3.0], name = "a", shape = (2,0), dtype = "float64", verify_shape = "true")
print (a)

#%%

# zeros

a = tf.zeros([2,2], tf.float32)

b = tf.zeros_like(a, optimize = True)
with tf.Session() as sess:
    print (sess.run(a))
    print (sess.run(b))
    
#%%
    
random_num = tf.random_normal([2,3], mean = 1, stddev = 4,
                              dtype = tf.float32, seed = None, name = 'rnum')

with tf.Session() as sess:
    print (sess.run(random_num))
    
#%%
    
A = tf.Variable(3, name = "number")
B = tf.Variable([1,3], name = "vector")
C = tf.Variable([[0,1], [2,3]], name = "matrix")
D = tf.Variable(tf.zeros([100]), name = "zero")
E = tf.Variable(tf.random_normal([2,3], mean = 1, stddev = 2, dtype = tf.float32))
    
#%%
# ???
# Variables initialization

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
#%%    
# Initialize subset
    
init_subset = tf.initialize_variables([a,b], name = "init_subset")
with tf.Session() as sess:
    sess.run[init_subset]

# Initialize single values
#%%
init_var = tf.Variable(tf.zeros([2,5]))
with tf.Session() as sess:
    sess.run(init_var.initializer)

# ???
    
#%%

# Model save

var1 = tf.Variable([1,3], name = "v1")
var2 = tf.Variable([2,4], name = "v2")

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    save_path = saver.save(sess, "test/save.ckpt")
