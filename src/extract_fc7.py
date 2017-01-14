import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import argparse
import utils
import numpy as np
import pickle
import time
import os

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                   help='train/val')
    parser.add_argument('--model_path', type=str, default='Data/vgg16.tfmodel',
                   help='Pretrained VGG16 Model')
    parser.add_argument('--data_dir', type=str, default='Data',
                   help='Data directory')
    parser.add_argument('--batch_size', type=int, default=1,
                   help='Batch Size')



    args = parser.parse_args()
    
    vgg_file = open(args.model_path)
    vgg16raw = vgg_file.read()
    vgg_file.close()
    
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(vgg16raw)
    
    print ("VGG done successfully")

    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })

    graph = tf.get_default_graph()

    for opn in graph.get_operations():
        print "Name", opn.name, opn.values()

    #image_id_list = [img_id for img_id in image_ids]
    image_names = os.listdir('./Images/')
    image_names.sort()
    image_id_list = []

    for i in range(len(image_names)):
        image_id_list.append(i)

    print "Total Images", len(image_id_list)
    
    
    sess = tf.Session()
    fc7 = np.ndarray( (len(image_id_list), 4096 ) )
    idx = 0

    from_start = time.clock()

    while idx < len(image_id_list):
        start = time.clock()
        image_batch = np.ndarray( (args.batch_size, 224, 224, 3 ) )

        count = 0
        for i in range(0, args.batch_size):
            if idx >= len(image_id_list):
                    break
            #image_file = join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
            image_file = "Images/"+image_names[idx]
            image_batch[i,:,:,:] = utils.load_image_array(image_file)
            idx += 1
            count += 1
        
        
        feed_dict  = { images : image_batch[0:count,:,:,:] }
        fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
        fc7_batch = sess.run(fc7_tensor, feed_dict = feed_dict)
        fc7[(idx - count):idx, :] = fc7_batch[0:count,:]
        end = time.clock()
        print "Time for batch 1 photos", end - start
       # print "Hours For Whole Dataset" , (len(image_id_list) * 1.0)*(end - start)/60.0/60.0/10.0
        print "Time Elapsed:", (from_start)/60, "Minutes"

        print "Images Processed", idx

    np.savetxt('FC7_Features_Animation', fc7)

if __name__ == '__main__':
    main()
