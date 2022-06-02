import os
import numpy as np 
import cv2
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import h5py
import json



class Segmentation:
    
    def __init__(self,in_dir,set_indicadores):
        pass
    
    def otsu(self):

        with open(self.in_dir + "/configimage.json") as json_file:
            config = json.load(json_file)
        
        arr_train = config["train"]
        ar_test = config["test"]
        
        for direction in arr_train:
            for ind in range(0,len(self.set_indicadores)):
                 with h5py.File(direction, 'r') as f:
                    
                    image = f['x'][:,:,ind]
                    name = f['y'][()]
                    
                    save_path = self.in_dir + "/output/otsu/train" + "/" + self.set_indicadores[ind] + "/" + str(name)
                    
                    #print(" Shape:",image.shape)
                    
                    #print(name)
                    
                    _,image_seg = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cv2.imwrite(save_path,image_seg)
        
        pass
    
    def kmeans(self): 
        pass
    
    

    
    
    




