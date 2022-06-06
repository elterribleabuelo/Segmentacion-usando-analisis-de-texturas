from scripts.preprocessing import Preprocessing
from scripts.featureExtraction import HaralickAndGabor
from scripts.utils import Generator


# =============================================================================
# Obtención de Frames
# =============================================================================

# frames_vaciado = Preprocessing(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
#                            "VaciadoPocket.wmv",1080,300,
#                            r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket\data\images\frames-Vaciado")

# frames_vaciado.getAndCropFrames()

# frames_llenado = Preprocessing(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
#                            "LlenadoDePocket.wmv",1080,300,
#                            r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket\data\images\frames-Llenado")

# frames_llenado.getAndCropFrames()


# =============================================================================
# Extracción de caracteríticas
# =============================================================================

features = HaralickAndGabor(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
                            r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket\data\images",
                            15,
                            (25,25))
features.FeaturesHaralick()


# =============================================================================
# Generar data en formato .h5 
# =============================================================================

prueba = Generator(r"C:\Users\titos\Github\Proyecto CV - Analisis de vaciado bucket",
                   r"data\images",
                   r"data\dataset-h5",
                   r"data\dataset-image-h5",
                   1,
                   ['dissimilarity','energy','homogeneity'],
                   0.70,
                   r"data\images-indicadores-haralick")

prueba.imagetoDataframe()
prueba.split_dataset(1)
prueba.otsu_segmentation()

asd = prueba.split_dataset()

asd[1]


prueba.getShapeImage()
names = prueba.getNamesImages(modo = 0)

print(names)

names[0]

names[1]
prueba.getNamesImages()

ss = prueba.imagetoDataframe()

ss.shape

ss.head()


# =============================================================================
# 
# =============================================================================
