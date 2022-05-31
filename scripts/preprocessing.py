import os
import cv2
import imutils
import numpy as np
import pandas as pd


class Preprocessing:
    
    def __init__(self,in_dir,video,width_show,frecuencia,out_dir,counter = 0):
        self.in_dir = in_dir
        self.video = video
        self.width_show = width_show
        self.frecuencia = frecuencia
        self.out_dir = out_dir
        self.counter = counter
    
    def getAndCropFrames(self):
        cap = cv2.VideoCapture(self.in_dir + "/data/videos/" + str(self.video))
        while True:
            ret,frame = cap.read()
            #print(frame.shape) # 1028,1920,3
            if ret == False:
                break
            frame_show = imutils.resize(frame,self.width_show)
            crop_frame = frame[50:920,370:1500,:] # 50:900,400:1500,:
            if (self.counter % int(self.frecuencia) == 0): # saca una imagen cada 10 segundos si es que frecuencia = 100
                cv2.imwrite(os.path.join(self.in_dir , 
                                         "data/images", 
                                         self.out_dir, str(self.video[:-4]) + str("_") + str(int(self.counter) // int(self.frecuencia)) + str('.png')),
                                         crop_frame)
            self.counter = self.counter + 1
            cv2.imshow('frame', frame_show)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
            elif k == 32:
                cv2.waitKey()
        
        cap.release()
        cv2.destroyAllWindows()