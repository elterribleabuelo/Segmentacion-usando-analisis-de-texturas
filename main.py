from scripts.preprocessing import Preprocessing
from scripts.featureExtraction import HaralickAndGabor


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
# Ejecutar modelo
# =============================================================================
