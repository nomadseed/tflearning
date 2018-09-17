# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:06:21 2018

!!! PERSONAL USE ONLY !!!

The topic is a sub of the forward leading car detection project.
assume we have inputs as: 
    many bounding boxes [x, y, width, height]
    two classes {'sideways':0, 'leading':1}
    loss funtion: cross entropy

objective:
    build a light-weighted nerual network, train it, save the event, checkpoint,
    pipeline(if possible) and a frozengraph in pb format
    
Note:
    1. don't run the script while running a tensorboard under CWD
    2. every tensor should be under tf.variable_scope('XXX',reuse=tf.AUTO_REUSE)
        if debugging the network with spyder, for as default, the spyder won't 
        allow reusing same piece of RAM for different tensor
        
    
@author: Wen Wen
"""

import json
import os
#import cv2
import tensorflow as tf

from numpy.random import RandomState

class Tiny_CNN():
    '''
    a customized tiny cnn just for simple classification works
    default dimension of the data is 4, learning rate 0.001
    
    '''
    def __init__(self, dim=4, learning_rate=0.001):
        ###########################  Part I: build a tensor graph #####################
        with tf.variable_scope('TinyCNN',reuse=tf.AUTO_REUSE):
            # use a variable scope to specify some certain specs for those 
            # variables.   
            ''' variables & conv kernels'''
            self.w1=tf.get_variable('w1',initializer=tf.random_normal([dim,dim],stddev=1,seed=1))
            self.w2=tf.get_variable('w2',initializer=tf.truncated_normal([dim,1],stddev=1,seed=1))
            
            ''' placeholder for inputs '''
            self.x=tf.placeholder(tf.float32,shape=(None,dim),name='x') # number of input, demonsion
            self.y=tf.placeholder(tf.float32,shape=(None,1),name='y')
            
            ''' define the layers '''
            self.y1=tf.nn.relu(tf.matmul(self.x,self.w1),name='y1')
            self.y_out=tf.nn.relu(tf.matmul(self.y1,self.w2),name='y_out')
            
            ''' define training step '''
            # take cross entropy as loss function
            self.cross_entropy=-tf.reduce_mean(self.y*tf.log(tf.clip_by_value(self.y_out,1e-10,1.0)),name='minus_cross_entropy') 
            # minimize the loss using Adam Optimizer
            self.train_step=tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

def Def_Input(filepath=None):   
    ''' 
    define the input, 
    use randomly generated data and label for debugging, if filepath== None
    
    '''
    if filepath == None:
        rdm=RandomState(1) # use a certain seed number to keep input unchanged 
        data_size=5000
        data=rdm.rand(data_size,4) 
        label = [[int(x1+x2+x3+x4<2.0)] for (x1,x2,x3,x4) in data] 
        
    return data, label

def Sess_Summaries(cnn):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('loss'):
    # define trianing loss
    training_loss = cnn.cross_entropy
    # save current state of training loss into summary variable
    tf.summary.scalar('training_loss', training_loss)


##################### Part II: run the session, do evaluation #################
    
if __name__ == "__main__":
    ''' define the optional arguements'''
    
    ''' prepare input '''
    batch_size=10
    train_data, train_label = Def_Input()
    data_size=len(train_data)
    
    ''' clear previous graph in memory '''
    # Clears the default graph stack and resets the global default graph.
    # this is especially useful while debugging
    tf.reset_default_graph()
    
    ''' initiate a network '''
    tinycnn=Tiny_CNN()
    Sess_Summaries(tinycnn)
    
    ''' initialize a session '''
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ''' run session '''
        ''' save the tensor graph (train)'''
        train_writer = tf.summary.FileWriter('./ckpt', sess.graph)
        merged = tf.summary.merge_all()
        ckpt_saver = tf.train.Saver()
        ckpt_saver.save(sess, './ckpt/model', global_step=0) # original status
        
        max_steps=10001
        for step in range(max_steps):
            # choose start and end point, make sure do the calculation in a 
            # single epoch if step is too large, training epoches repeatly
            start = step * batch_size % data_size 
            end = min(start + batch_size,data_size)
            
            ''' training '''
            # training steps
            # it's observed that if we put a [start:end] after the data and 
            # label array, it will process way faster than not putting it
            summary, _ = sess.run([merged,tinycnn.train_step],feed_dict={tinycnn.x:train_data[start:end],tinycnn.y:train_label[start:end]})
            
            # save training result per 200 steps
            if step % 1000 == 0:
                # when calculate the loss, never put a [start:end] after data and label
                train_loss = sess.run(tinycnn.cross_entropy,feed_dict={tinycnn.x:train_data,tinycnn.y:train_label})
                print("iteration %d, training loss %g" %(step,train_loss))
                train_writer.add_summary(summary, step)

            ''' validation '''
            
                
            
        # close file writer and session
        ckpt_saver.save(sess, './ckpt/model', global_step=max_steps) # final status
        train_writer.close()

    
    print("Porgram ended for reaching maximum training steps")
    
    
''' End of file '''