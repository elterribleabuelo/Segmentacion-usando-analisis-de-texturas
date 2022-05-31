import os
import numpy as np 
import cv2
#import sklearn.cluster._k_means as Kmeans


class Model:
    
    def __init__(self,in_dir,images_dir,set_indicadores,indicadores_dir):
        self.in_dir = in_dir # C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket
        self.images_dir = images_dir # data\images
        self.set_indicadores = set_indicadores
        self.indicadores_dir = indicadores_dir # data\images-indicadores-haralick

    
    def getNamesImages(self):
        
        frames_Llenado = []
        frames_Vaciado = []
        frames_All = []
        count = 0
        
        path_dir_main = self.in_dir + str("/") + self.images_dir
        
        for directorio in os.listdir(path_dir_main):
            
            path_tip_frame = path_dir_main + "/" + directorio
            
            # print(path_tip_frame)
            
            for image in os.listdir(path_tip_frame):
                if count == 0:
                    frames_Llenado.append(image)
                else:
                    frames_Vaciado.append(image)
            
            count += 1
        
        frames_All.append(frames_Llenado)
        frames_All.append(frames_Vaciado)
        
        return frames_All
    
    def imagetoarray(self):
        
        names_frames = self.getNamesImages()
        
        path_dir = self.in_dir + str("/") +self.indicadores_dir
        
        count = 0
        
        for directorio in os.listdir(path_dir):
                        
            path_directorio = path_dir + "/" + directorio
            print("Path directorio:",path_directorio)
            
            for frame in names_frames[count]:
                
                for indicador in self.set_indicadores:
                    
                    image_dir = path_directorio + "/" + indicador + "/" + frame
                    image = cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE)
                    
            
            count += 1
        
        
        pass
    
    def clustering(self):
        pass
    

prueba = Model(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
               r"data\images",
               ['ASM', 'correlation','contrast','dissimilarity','energy','homogeneity'],
               r"data\images-indicadores-haralick")

names = prueba.getNamesImages()

print(names)

names[0]

names[1]
prueba.getNamesImages()

ss = prueba.imagetoarray()
    