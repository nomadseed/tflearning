# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:31:29 2018

learning tensorflow from beginning

tensorflow framework has two parts, one is tensor graph, the other is running
sessions, during which we often evaluate the session result and make an end.

The topic is a sub of the forward leading car detection project.
assume we have inputs as: 
    many bounding boxes [x, y, width, height]
    two classes {'sideways':0, 'leading':1}
    loss funtion: cross entropy

objective:
    build a light-weighted nerual network, train it, save the event, checkpoint,
    pipeline(if possible) and a frozengraph in pb format

@author: Wen Wen
"""

# Imports
import json
import os
#import cv2
import tensorflow as tf

from numpy.random import RandomState

###########################  Part I: build a tensor graph #####################

''' constant '''
#c1 = tf.constant(3.0)
#c2 = tf.constant(2.0)

''' simple calculation '''
#add=tf.add(c1,c2,name='Add') # tf.subtract
#mul=tf.multiply(c1,c2,name='Multiply')
#div=tf.divide(c1,c2,name='Divide') # devide function needs float32 input
#mod=tf.mod(c1,c2,name='Mod')
#other function like: tf.abs, tf.argmax, tf.atan, tf.exp, tf.log, etc
#print(add,mul,div,mod)

''' variables & conv kernels'''
# note that tf.variable is a function, tf.Variable is a class, here we use the 
# class
w1=tf.Variable(tf.random_normal([10,12],stddev=1,seed=1),name='w1') # old version
#w1=tf.get_variable('w1',initializer=tf.random_normal([3,3],stddev=1,seed=1))
# standard deviation 1, generating seed 1. usually the seed is for making 
# comparison among different implementations
w2=tf.Variable(tf.truncated_normal([12,1],stddev=1,seed=1),name='w2') # old version
#w2=tf.get_variable('w2',initializer=tf.truncated_normal([3,1],stddev=1,seed=1))
# truncated normal function preserve the value in between [mean-2*stddev,mean+2*stddev] only
#print(w1,w2)

''' placeholder '''
# the placeholder is for the later input given from optional arguements, it will
# only create a RAM without values, the values will be fed during session
x=tf.placeholder(tf.float32,shape=(None,10)) # number of input, demonsion
y=tf.placeholder(tf.float32,shape=(None,1))

''' define the layers '''
# for layer operation, we have conv2d/3d, matmul and other possible operators
y1=tf.nn.relu(tf.matmul(x,w1))
y_out=tf.nn.relu(tf.matmul(y1,w2))
# for activating functions, we have relu. etc

''' define training step '''
# take cross entropy as loss function
cross_entropy=-tf.reduce_mean(y*tf.log(tf.clip_by_value(y_out,1e-10,1.0))) 
# minimize the loss using Adam
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

''' define the input '''
# usually input can be either specified here or taken as optional arguements
rdm=RandomState(1) # use a certain seed number to keep input unchanged 
data_size=5000
batch_size=10
X=rdm.rand(data_size,10) # pseudo input
Y = [[int(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10<5.0)] for (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) in X] # pseudo label as benchmark




##################### Part II: run the session, do evaluation #################


if __name__ == "__main__":
    ''' define the optional arguements'''
    
    ''' initialization '''
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ''' run session '''
        ''' save the tensor graph '''
        graph_writer = tf.summary.FileWriter('./ckpt', sess.graph)
        ckpt_saver = tf.train.Saver()
        ckpt_saver.save(sess, './ckpt/model', global_step=0) # original status
        
        ''' check the random filter '''
        #print(sess.run(w1))
        #print(sess.run(w2))
        
        steps=10001
        for i in range(steps):
            #choose start and end point, make sure do the calculation in a single epoch
            start = i * batch_size % data_size # if step is too large, training epoches repeatly
            end = min(start + batch_size,data_size)
            sess.run(train_step,feed_dict={x:X[start:end],y:Y[start:end]})
            if i % 200 == 0:
                training_loss= sess.run(cross_entropy,feed_dict={x:X,y:Y})
                print("iteration %d, training loss %g" %(i,training_loss))
        
        # close file writer and session
        ckpt_saver.save(sess, './ckpt/model', global_step=steps) # final status
        graph_writer.close()


    ''' 
    the other way of doing session
    sess = tf.Session()
    print(sess.run(c))
    sess.close()
    '''
    
    print("Porgram ended")

""" End of the file """