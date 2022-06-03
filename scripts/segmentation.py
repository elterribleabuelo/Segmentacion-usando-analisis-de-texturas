import os
import numpy as np 
import cv2
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import h5py
import json
import scipy



class Segmentation:
    
    def __init__(self,in_dir,set_indicadores,num_cluster,alpha):
        
        self.in_dir = in_dir
        self.set_indicadores = set_indicadores
        self.num_cluster = num_cluster
        self.alpha = alpha
        
    
    def otsu(self):
        """
        Función que se encarga de hacer la segmentación de la imagen por la umbralización de Otsu.
        Se toma como entrada las imágenes después de aplicar los indicadores de Haralick.
        Una misma imagen tiene "bandas" las cuales representan los indicadores que se detallan 
        en el paraámetro set_indicadores.

        Returns
        -------
        None.

        """

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
    
    def compute_dictionary_one_image(self, alpha, response):
        """
        

        Parameters
        ----------
        response : TYPE
            DESCRIPTION.

        Returns
        -------
        alphaed_response : TYPE
            DESCRIPTION.

        """
   
        d = response.shape[0]*response.shape[1]
        response = response.reshape((d,-1)) 
        alphas = np.random.choice(d, alpha, False)
        alphaed_response = response[alphas]
   
        return alphaed_response
    
    def kmeans_compute_dictionary(self):
        """
        Funcion que se encarga de realizar el entrenamiento del algoritmo
        de Kmeans 

        Returns
        -------
        dictionary : np.array
            DESCRIPTION.Contiene los centros de los cluster hallados

        """
        
        with open(self.in_dir + "/configimage.json") as json_file:
            config = json.load(json_file)
        
        arr_train = config["train"]
        #ar_test = config["test"]

        m = []
        
        for direction in arr_train:
            with h5py.File(direction, 'r') as f:
                    
                    image = f['x'][:]
                    #name = f['y'][()]
                    re = self.compute_dictionary_one_image(self.alpha, image)
                    m.append(re)
            
            
        m = np.array(m)
        n = m.shape[0]*m.shape[1]
        final_response = m.reshape((n,-1))
        
        kmeans = KMeans(n_clusters = self.num_cluster).fit(final_response)
        dictionary = kmeans.cluster_centers_
        
        return dictionary
    
    def get_visual_words(self,path_image,dictionary):
        """
        

        Parameters
        ----------
        path_image : TYPE
            DESCRIPTION.
        dictionary : TYPE
            DESCRIPTION.

        Returns
        -------
        visual_words : TYPE
            DESCRIPTION.

        """
        
        
        
        img = cv2.imread(path_image)
        
        response=img.reshape(img.shape[0]*img.shape[1],-1)
        
        #dictionary = self.kmeans_compute_dictionary()
         
        dist = scipy.spatial.distance.cdist(response, dictionary)
         
        visual_words = np.argmin(dist, axis=1)
        visual_words = visual_words.reshape(img.shape[0],img.shape[1])
    
        return visual_words

    
seg = Segmentation(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
                   ['dissimilarity','energy','homogeneity'],
                   2,
                   20
                   )

centers = seg.kmeans_compute_dictionary()


image_visual = seg.get_visual_words(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket\data\images\frames-Vaciado\VaciadoPocket_45.png",centers)

image_visual = image_visual.astype(np.uint8)*255

cv2.namedWindow("Prueba", cv2.WINDOW_NORMAL)

cv2.imshow("Prueba", image_visual)

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
#closing all open windows 
cv2.destroyAllWindows() 

# path = r'C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket\data\dataset-image-h5\frames-Vaciado'
# path_h5 = path + "/VaciadoPocket_53.h5"

# with h5py.File(path_h5, 'r') as f:
    
#     image = f['x'][:]
#     name = f['y'][()]
    

# alpha = 10
# d = image.shape[0]*image.shape[1]
# print(d)
# response = image.reshape((d,-1)) 
# print(response.shape)
# alphas = np.random.choice(d, alpha, False)

# print(alphas)

# alphaed_response = response[alphas]

# print(alphaed_response.shape) # (alpha,len(self.set_indicadores))


