from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Aggiungiamo la radice al sistema di ricerca di Python
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils import DIR_DATA, DIR_PROJECT

def extract_aligned_lego(img, obb_box, padding_factor=1.1):
    
    #Estrae un crop raddrizzato con un margine extra.
    #padding_factor=1.1 significa il 10% in più di spazio intorno.
    
    x_c, y_c, w, h, angle_rad = obb_box
    angle_deg = np.degrees(angle_rad)

    # Calcolo dimensioni con padding
    # Moltiplichiamo larghezza e altezza per il fattore di padding
    w_padded = w * padding_factor
    h_padded = h * padding_factor

    # Area di sicurezza per la rotazione
    # Usiamo la diagonale del rettangolo PADDATO per non tagliare gli angoli
    diag = int(np.sqrt(w_padded**2 + h_padded**2))
    
    x1, y1 = int(x_c - diag/2), int(y_c - diag/2)
    x2, y2 = int(x_c + diag/2), int(y_c + diag/2)

    img_h, img_w = img.shape[:2]
    temp_crop = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
    
    if temp_crop.size == 0: return None

    # Rotazione locale
    (th, tw) = temp_crop.shape[:2]
    t_center = (tw // 2, th // 2)
    M = cv2.getRotationMatrix2D(t_center, angle_deg, 1.0)
    rotated_temp = cv2.warpAffine(temp_crop, M, (tw, th))

    # Ritaglio finale PADDATO
    # Estraiamo (w_padded, h_padded) invece di (w, h)
    final_crop = cv2.getRectSubPix(rotated_temp, (int(w_padded), int(h_padded)), t_center)
    
    return final_crop


# Carica il modello
model = YOLO(str(DIR_PROJECT / "models/yolo_labeling_alpha/weights/best.pt"))

data_path = DIR_DATA / "detection" / "images"
data_path_img = data_path / "img"
data_path_crop = data_path / "crop"


for img_path in data_path_img.iterdir():
    results = model.predict(source=img_path, conf=0.6)

    img_orig = cv2.imread(img_path)

    for r in results:
        # I risultati OBB sono in r.obb
        if r.obb is not None:
            # xywhr: centro_x, centro_y, width, height, rotation (in radianti)
            for j, box in enumerate(r.obb.xywhr.cpu().numpy()): # type: ignore
                
                # Creiamo il rettangolo per il ritaglio dritto (Bounding Box orizzontale)
                # Aggiungiamo un piccolo margine per non tagliare i bordi
                crop = extract_aligned_lego(img_orig, box)
                
                if crop and crop.size > 0:
                    cv2.imwrite(f"{data_path_crop}/crop_lego_{j}.jpg", crop) # type: ignore
                    print(f"✅ Crop {j} salvato con successo.")