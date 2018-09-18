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
    3. to get a correct ckpt file, restart the IDE kernel if needed
        
    
@author: Wen Wen
"""

import json
import os
import argparse
import numpy as np
#import cv2
import tensorflow as tf

from numpy.random import RandomState

class Tiny_CNN():
    '''
    a customized tiny cnn just for simple classification works
    default dimension of the data is 4, learning rate 0.001
    
    '''
    def __init__(self, dim=4, learning_rate=0.0001):
        ###########################  Part I: build a tensor graph #####################
        with tf.variable_scope('TinyCNN',reuse=tf.AUTO_REUSE):
            # use a variable scope to specify some certain specs for those 
            # variables.   
            
            ### define the global parameters for
            self.global_step=tf.get_variable('global_step',initializer=tf.random_normal([1],stddev=1,seed=1))
            
            ### variables & conv kernels
            self.w1=tf.get_variable('w1',initializer=tf.random_normal([dim,dim],stddev=1,seed=1))
            self.w2=tf.get_variable('w2',initializer=tf.truncated_normal([dim,dim],stddev=1,seed=1))
            self.w3=tf.get_variable('w3',initializer=tf.truncated_normal([dim,2],stddev=1,seed=1))
            
            ### placeholder for inputs
            self.x=tf.placeholder(tf.float32,shape=(None,dim),name='x') # number of input, demonsion
            self.y=tf.placeholder(tf.float32,shape=(None,2),name='y')
            
            ### define the layers
            self.y1=tf.nn.relu(tf.matmul(self.x,self.w1),name='y1')
            self.y2=tf.nn.relu(tf.matmul(self.y1,self.w2),name='y2')
            self.y_out=tf.nn.relu(tf.matmul(self.y2,self.w3),name='y_out')
            
            ### define training step and loss 
            # cross entropy from tensorflow
            self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.y, logits = self.y_out, name='cross_entropy')
            # put customized loss function here
            self.loss=tf.add(self.cross_entropy[0],self.cross_entropy[1],name='loss')
            # minimize the loss using Adam Optimizer
            # the default AdamOptimizer is used for large dataset, thus the beta
            # values are large (default beta1=0.9, beta2=0.999). tune down the 
            # beta values if running a smaller dataset (e.g. beta1=0.7,beta2=0.9)
            self.train_step=tf.train.AdamOptimizer(learning_rate,beta1=0.9,beta2=0.93).minimize(self.cross_entropy)
            
            
    def get_global_step(self,globalstep):
        self.global_step=globalstep

            
def Load_json_anno(filepath,jsonlabel='annotationfull',foldernumber=float('Inf')):
    '''
    load annotation from json files
    args:
        filepath: path of file
        jsonlabel: a unique label in file name to locate the json file
        foldernumber: how many subfolders to search
        
    '''
    annotationdict={}
    folderdict=os.listdir(filepath)
    foldercount=0
    
    for folder in folderdict:
        # skip the files, choose folders only
        if '.' in folder:
            continue 
        
        # for debug, set the number of folders to be processed
        if foldercount>=foldernumber:
            break
        else:
            foldercount+=1
            
        imagepath=os.path.join(filepath,folder)
        filedict=os.listdir(imagepath)
            
        for jsonname in filedict:
            if 'json' in jsonname and jsonlabel in jsonname:
                annotations=json.load(open(os.path.join(imagepath,jsonname)))
                annotationdict[folder]=annotations
     
    return annotationdict

def Get_label_from_class(classname):
    '''
    get label (number) according to the class name of leading/sideways car
    
    one hot label for binary classification as ['sideways & opposite', 'leading']
    '''
    val1 = 0
    val2 = 1
    if classname == 'sideways' or classname == 'opposite':
        val1 = 1.
        val2 = 0.
    elif classname == 'leading':
        val1 = 0.
        val2 = 1.
    else:
        raise ValueError('wrong class name in annotation')
    return [val1,val2]

def Normalize_data(x,y,width,height,imgwidth=640,imgheight=480):
    '''
    to use cross entropy as our loss function, we normalize the data from
    [0,640] and [0,480] to [0,1]
    
    '''
    x_ = float(x)/imgwidth
    y_ = float(y)/imgheight
    width_ = float(width)/imgwidth
    height_ = float(height)/imgheight
    return [x_, y_, width_, height_]
    
def Extract_data_and_label(annotationdict,normalize_flag=True):
    '''
    extract data and label from annotation
    note that the data returned is in form of ndarray
    
    '''
    data=[]
    label=[]
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            annotationlist=annotationdict[foldername][imagename]['annotations']
            for i in annotationlist:
                if None in [i['x'],i['y'],i['width'],i['height']]:
                    continue
                if normalize_flag:
                    data.append(Normalize_data(i['x'],i['y'],i['width'],i['height']))
                else:
                    data.append([i['x'],i['y'],i['width'],i['height']])
                label.append(Get_label_from_class(i['category']))
                
    return np.array(data), np.array(label)

def Def_input(filepath=None):   
    ''' 
    define the input, 
    use randomly generated data and label for debugging, if filepath== None
    
    '''
    if filepath == None:
        rdm=RandomState(1) # use a certain seed number to keep input unchanged 
        data_size=5000
        data=rdm.rand(data_size,4) 
        label = [[int(x1+x2+x3+x4<2.0)] for (x1,x2,x3,x4) in data] 
    else:
        annotationdict = Load_json_anno(filepath)
        data, label = Extract_data_and_label(annotationdict)
    return data, label, annotationdict

def Sess_summaries(cnn):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('training_loss'):
        # define trianing loss
        training_loss = cnn.loss

        # save current state of training loss into summary variable
        tf.summary.scalar('training_loss', training_loss)

    
##################### Part II: run the session, do evaluation #################
    
if __name__ == "__main__":
    ### Define the optional arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/EventFrames', 
        help="select the file path for image folders")
    args = parser.parse_args()
    filepath=args.file_path
    
    ### Prepare input 
    batch_size=10
    train_data, train_label, _ = Def_input(filepath)
    data_size=len(train_data)
    
    ### Clear previous graph in memory
    # clears the default graph stack and resets the global default graph.
    # this is especially useful while debugging
    tf.reset_default_graph()
    
    ### initiate a network
    tinycnn=Tiny_CNN()
    Sess_summaries(tinycnn)

    ### initialize a session
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ### run session
        # save the tensor graph (train)
        train_writer = tf.summary.FileWriter('./ckpt', sess.graph)
        merged = tf.summary.merge_all()
        ckpt_saver = tf.train.Saver()
        ckpt_saver.save(sess, './ckpt/model', global_step=0) # original status
        
        max_steps=200000
        for step in range(max_steps):
            # choose start and end point, make sure do the calculation in a 
            # single epoch if step is too large, training epoches repeatly
            start = step * batch_size % data_size 
            end = min(start + batch_size,data_size)
            
            ### training
            # training steps
            # it's observed that if we put a [start:end] after the data and 
            # label array, it will process way faster than not putting it
            summary, _ = sess.run([merged,tinycnn.train_step],feed_dict={tinycnn.x:train_data[start:end],tinycnn.y:train_label[start:end]})
            
            # save training result per 200 steps
            if step % 200 == 0:
                # when calculate the loss, never put a [start:end] after data and label
                train_loss = sess.run(tinycnn.loss,feed_dict={tinycnn.x:train_data,tinycnn.y:train_label}) 
                train_writer.add_summary(summary, step)
            
            # print training result per 5000 steps
            if step % 5000 ==0:
                print("iteration %d, training loss %g" %(step,train_loss))
            
            ### validation 
                
            
        # close file writer and session
        ckpt_saver.save(sess, './ckpt/model', global_step=max_steps) # final status
        train_writer.close()

    
    print("train and val ended for reaching maximum steps")
    
    
''' End of file '''