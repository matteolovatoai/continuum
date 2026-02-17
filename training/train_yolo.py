from pathlib import Path
import sys
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Aggiungiamo la radice al sistema di ricerca di Python
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils import DIR_MODELS

# 1. Carichiamo il modello small pre-addestrato per OBB
model = YOLO(DIR_MODELS / "yolov8s-obb.pt") 

# 2. Avviamo il training
results = model.train(
    name="yolo_labeling_alpha", # nome con cui salvo il modello
    project=DIR_MODELS,           # cartella dove salvo il progetto
    data="yolo_labeling_alpha.yaml",    # dove sono i dati? guarda dentro il file yaml
    epochs=100,             # Con poche foto, servono un po' di epoche
    imgsz=640,              # Standard
    device="mps",           # GPU del Mac
    batch=8,                # Batch piccolo per 30 immagini
    plots=True,             # Genera i grafici per vedere se impara
    mixup=0.1,              # Aiuta a gestire le sovrapposizioni
    hsv_h=0.015,            # Varia leggermente il colore
    hsv_v=0.6,              # Aumenta la variazione di luminosità (Value)
    hsv_s=0.5,              # Varia la saturazione
    degrees=180,
    flipud=0.5,             # Capovolge l'immagine (50% probabilità)
    fliplr=0.5,             # Specchia l'immagine (50% probabilità)
)

'''path_onnx = model.export(
    format="onnx", 
    imgsz=640, 
    dynamic=True, 
    simplify=True
)'''