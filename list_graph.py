# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:13:32 2019

check the graph and list all the tensors in the graph

@author: Wen Wen
"""
import tensorflow as tf
from tensorflow.python.framework import tensor_util
import numpy as np
import argparse


def showGraphMembers(img_w=300,img_h=300):
    """
    show the graph members
    
    """
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_name_dict={name:name for name in all_tensor_names}
    tensor_dict={name:tf.get_default_graph().get_tensor_by_name(name) for name in all_tensor_names}
    output_tensor = {}
    for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            output_tensor[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    
    # create a white image
    image_np=np.ndarray((img_h, img_w, 3)).astype(np.uint8)

    # update the value of entire graph once    
    with tf.Session() as sess:
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        output_dict= sess.run(output_tensor,feed_dict={image_tensor:np.expand_dims(image_np, 0)})
    
    
    #get value of a tensor
    test_tensor=tf.get_default_graph().get_tensor_by_name('FeatureExtractor/MobilenetV2/expanded_conv_1/depthwise/depthwise_weights/read:0')
    print(test_tensor)
    
    return tensor_name_dict

def showOperations(graph, show=False):
    ops = graph.get_operations()
    ops_dict={}
    type_dict=[]
    for op in ops:
        if show==True:
            print('- {0:20s} "{1}" ({2} outputs)'.format(op.type, op.name, len(op.outputs)))
        opdict={}
        opdict['name']=op.name
        opdict['type']=op.type
        if op.type not in type_dict:
            type_dict.append(op.type) 
        opdict['output_len']=len(op.outputs)
        ops_dict[op.name]=opdict
    
    return ops_dict, type_dict

def showGraphNodes(od_graph_def, const_only=True):
    
    graph_nodes=[n for n in od_graph_def.node]
    wts_nd={}
    if const_only:
        wts = [n for n in graph_nodes if n.op=='Const']
        for n in wts:
            wts_nd[n.name]=tensor_util.MakeNdarray(n.attr['value'].tensor)
    else:
        for n in graph_nodes:
            if n.op=='Const':
                wts_nd[n.name]=tensor_util.MakeNdarray(n.attr['value'].tensor)
            else:
                wts_nd[n.name]=n.attr['value'].tensor
    
    
    return graph_nodes, wts_nd
    
if __name__=='__main__':
    # pass the parameters
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/FrameImages
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/tf-object-detection-api/research/viewnyx/ckpt_ssd_opt_300/export/frozen_inference_graph.pb', 
                        help="select the file path for ckpt folder")
    args = parser.parse_args()
    
    ckptpath = args.ckpt_path
    
    # Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    # initialize the graph once for all
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckptpath, 'rb') as fid:
            serialized_graph = fid.read()
            # the od_graph_def is the config file for the network
            # when display this variable, please wait, cause it read the file in binary
            # but display with utf8
            
            # there is a NonMaximumSupressionV3 in the graph, but only V1 & V2 given
            # in the api if tf version<1.8.0
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
            # list graph nodes
            _, wts_nd = showGraphNodes(od_graph_def, 
                                                      const_only=True)
            
            # list graph members and show (including sessions)
            #tensor_name_dict = showGraphMembers()
            
            # list graph operations
            ops_dict, type_dict = showOperations(detection_graph,show=False)
            
            # for debugging
            #tensor=wts_nd['FeatureExtractor/MobilenetV2/expanded_conv_2/depthwise/depthwise_weights']
            #tensor_nd=tensor[:,:,:,0]
            
            
            
    
""" End of file """
