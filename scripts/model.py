import os
import numpy as np 
import cv2
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import h5py



class Model:
    
    def __init__(self,in_dir,images_dir,h5_dir,set_indicadores,indicadores_dir):
        self.in_dir = in_dir # C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket
        self.images_dir = images_dir # data\images
        self.h5_dir = h5_dir # data\dataset-h5
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
    
    
    def getShapeImage(self):
        
        names_frames = self.getNamesImages()
        
        path_dir = self.in_dir + str("/") +self.indicadores_dir
        
        count = 0
        
        for directorio in os.listdir(path_dir):
            path_directorio = path_dir + "/" + directorio
            for frame in names_frames[count]:
                for indicador in self.set_indicadores:
                    if count == 1:
                        break
                    
                    image_dir = path_directorio + "/" + indicador + "/" + frame
                    image = cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE)
                    
                    count += 1
        
        height = image.shape[0] # Numero de filas --> Y
        width = image.shape[1] # Numero de columnas --> X
        
        return (height,width)
        
    
    def getShapeDataframe(self):
        
        # Numero de imagenes
        all_images = self.getNamesImages()
        num_images = len(all_images[0]) + len(all_images[0])
        
        #print(num_images) # 292
        
        # Numero de pixeles por imagen
        height = self.getShapeImage()[0] # 870
        #print(height)
        width  =  self.getShapeImage()[1] # 1130
        #print(width)
        
        # Numero total de registros
        num_rows = height*width*num_images
        
        # Armando el dataframe 
        X = np.zeros((num_rows,len(self.set_indicadores)),dtype='uint8') # K x N muestras (filas), y Gab  características (columnas)
        columns_dataframe= [ str(i) for i in self.set_indicadores]
        
        # Dataframe final
        data = pd.DataFrame(X, columns = columns_dataframe)
        
        return data
    
    def imagetoDataframe(self):
        
        names_frames = self.getNamesImages()
        
        path_dir = self.in_dir + str("/") + self.indicadores_dir
        
        count = 0
        
        count_images = 0
        
        height = self.getShapeImage()[0]
        width  = self.getShapeImage()[1]
        
        num_pixels = height*width
        # print(num_pixels) --> 983100
        
        data = self.getShapeDataframe() # Dataset vacio
        #print(data)
        #print(data.shape)
        
        
        
        for directorio in os.listdir(path_dir):
                        
            path_directorio = path_dir + "/" + directorio
            
            path_dir_h5 = self.in_dir + str("/") + self.h5_dir + "/" + directorio
            #print("Path directorio:",path_directorio)
            
            for frame in names_frames[count]:
                
                # Declarar un contenedor que almacene las caracteristicas a estudiar de cada una de las imágenes
                # Debe tener la forma [abcdfg,len(self.set_indicadores)]
                # abcdefg : Debe tener como longitud image.shape[0]*image.shape[1]
                
                container_h5 = np.zeros((num_pixels,len(self.set_indicadores)), dtype = 'uint8')
                
                for (i,indicador) in enumerate(self.set_indicadores):
                    
                    image_dir = path_directorio + "/" + indicador + "/" + frame
                    image = cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE)
                    
                    # desde = count_images * num_pixels
                    # hasta = (count_images + 1) * num_pixels - 1
                    
                    # add_array = np.reshape(image,(image.shape[0]*image.shape[1],-1)).astype("uint8").tolist()
                    
                    # Error aqui
                    # Error aqui
                    # Error aqui
                    container_h5[:,i] = np.reshape(image,(image.shape[0]*image.shape[1],1)).astype("uint8").ravel()
                    
                with h5py.File(path_dir_h5 + "/" + frame[:-4] + ".h5", "w") as hdf:
                    
                   hdf.create_dataset('x', data = container_h5)
                   hdf.create_dataset('y', data = frame)
                    
                    
                    # Asignando al dataframe principal
                    # data.loc[desde:hasta,indicador] = add_array # vector columna
                    
                    #print("Fin de un indicador")
                
                print("Indicadores de la imagen, %s guardada correctamente." % (frame))                                   
                
                count_images += 1
                    
            
            count += 1
        
        
        return 0
    
    def clustering(self):
        pass
    

prueba = Model(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
               r"data\images",
               r"data\dataset-h5",
               ['dissimilarity','energy','homogeneity'],
               r"data\images-indicadores-haralick")


prueba.getShapeImage()
names = prueba.getNamesImages()

print(names)

names[0]

names[1]
prueba.getNamesImages()

ss = prueba.imagetoDataframe()

ss.shape

ss.head()



# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans

# https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type

# https://datascience.stackexchange.com/questions/44517/kmeans-large-dataset

# https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans