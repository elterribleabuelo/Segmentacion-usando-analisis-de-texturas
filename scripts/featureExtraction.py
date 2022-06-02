import os
import numpy as np
import cv2
from skimage.feature import greycomatrix,greycoprops


class HaralickAndGabor:
    
    def __init__(self,in_dir,images_dir,step_size,windowSize):
        
        self.in_dir = in_dir
        self.images_dir = images_dir
        self.step_size = step_size
        self.windowSize = windowSize
        #self.out_dir = out_dir
    
    def normalize_2d(self,matrix):
        # Only this is changed to use 2-norm put 2 instead of 1
        norm = np.linalg.norm(matrix, 1)
        # normalized matrix
        matrix = matrix/norm  
        return matrix
    
    def shape_imageSlidingWindows(self,matrix):
        
        count_X = 0 # numero de columnas
        count_Y = 0 # numero de filas
        
        for y in range(0,matrix.shape[0],self.step_size):
            for x in range(0,matrix.shape[1],self.step_size):
                count_X += 1
            count_Y += 1
        count_X = count_X // count_Y
        
        return (count_Y,count_X)
        
    def FeaturesHaralick(self):
        
        # Parámetros
        distances = [3] 
        angles = [-np.pi/4, 0, np.pi/4, np.pi/2] # (np.pi/2 --> (dx = 0 y dy = dst))
        
        # Características de Haralick 
        properties = ['ASM', 'correlation','contrast','dissimilarity','energy','homogeneity']
        
        for directorio in os.listdir(self.images_dir):
            
            for image in os.listdir(self.images_dir + str("/" + str(directorio))):
                
                
                # Imagen en escala de grises
                img = cv2.imread(self.images_dir + str("/" + str(directorio)) + "/" + str(image),cv2.IMREAD_GRAYSCALE)
                
                print("Dimensiones de la imagen original:",img.shape)
                
                #print(img.shape) (870, 1130)
                count_X = 0
                count_Y = 0
                
                # Contenedor de características de Haralick
                GLCMFeatures = np.zeros((self.shape_imageSlidingWindows(img)[0], 
                                         self.shape_imageSlidingWindows(img)[1], 
                                         len(properties)),dtype = np.float32)
                
                print("Dimensiones antes del resize:",GLCMFeatures.shape)
                
                # Sliding Windows
                for y in range(0,img.shape[0],self.step_size):
                    
                    count_X = 0
                    
                    for x in range(0,img.shape[1],self.step_size):
                    
                        # Seleccionando la porción de imagen a analizar
                        window_img = img[y:y + self.windowSize[1],x:x + self.windowSize[0]]
                        
                        # Matriz GLCM
                        
                        co_matrix = greycomatrix(window_img, distances = distances, angles = angles).astype('uint8')
                        #print(co_matrix.shape) # (256, 256, 1, 4)
                        dim = co_matrix.shape        
                        aux_co_matrix = np.zeros((dim[0],dim[1]))
                
                        # Nota: para esta ultima matriz sera de la forma (256,256,1,4)
                        
                        # Sumatoria de matrices
                
                        # Recorremos todas las matrices (en base a la distancia y angulo) 
                        # que se obtuvieron en co_matriz 
                        for distance in range(0,dim[2]):
                            for angle in range(0,dim[3]):
                                #print(distance,angle)
                                # Sumatoria que lo hace al descriptor invariante a la rotacion
                                aux_co_matrix += co_matrix[:,:,distance,angle] # Matriz sumatoria
                
                        # aux_co_matrix.shape = (256, 256) 
                        # NOTA: A ESTA MATRIZ DE FORMA (MXN) DEBO SACARLE LOS INDICADORES DE TEXTURA
                        
                        # Artificio 
                        _aux_co_matrix = aux_co_matrix[:, :, np.newaxis]
                        b = np.zeros((aux_co_matrix.shape[0],aux_co_matrix.shape[1]))
                        _b = b[:, :, np.newaxis]
                        _p = np.dstack((_aux_co_matrix , _b)) # _p.shape = (256,256,2)
                        __p = np.reshape(_p, (_p.shape[0],_p.shape[1],-1,_p.shape[2])) #__p.shape = (256,256,1,2)
                        
                        #print(count_Y,count_X)
                        # Sacamos los indicadores de Haralick
                        for i, prop in enumerate(properties):
                            GLCMFeatures[count_Y,count_X,i] =  greycoprops(__p, prop)[0,0] # escalar
                        count_X += 1
                    
                    
                    count_Y += 1
                
                print("Fin de extracción de características de la imagen:",image)
                for i,indicador in enumerate(properties):
                    sum_matrix_resized = cv2.resize(GLCMFeatures[:,:,i],
                                                              (img.shape[1],img.shape[0]), # (width, height)
                                                              interpolation = cv2.INTER_CUBIC)
                    sum_matrix_resized_norm = cv2.normalize(sum_matrix_resized, None, alpha = 0, 
                                                   beta = 255,norm_type = cv2.NORM_MINMAX, 
                                                   dtype = cv2.CV_32F).astype(np.uint8)
                    print("Dimensiones despues del resize:",sum_matrix_resized_norm.shape)
                    cv2.imwrite(os.path.join(self.in_dir,
                                             "data/images-indicadores-haralick",
                                             directorio,
                                             indicador,
                                             str(image)),sum_matrix_resized_norm)
                
                          