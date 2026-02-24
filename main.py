import base64
from contextlib import asynccontextmanager
from tabnanny import verbose
from time import sleep
import threading
import cv2
from fastapi import FastAPI, Response
from ultralytics import YOLO

from src.utils import DIR_YOLO_ALPHA
# carico yolo solo la prima volta
model = YOLO(str(DIR_YOLO_ALPHA))
# dati di riferimento dell'ultimo frame valido
data_stored = {"status": "start"}
camera = cv2.VideoCapture(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # aspetta finchè in esecuzione
    yield
    # quando premi ctrl+c per chiudere allora:
    camera.release()
    cv2.destroyAllWindows()

app = FastAPI(lifespan=lifespan)

# catturare frame e elaborarlo e salvare i dati sulla variabile globale
def update_frame():
    while True:
        ret, frame = camera.read()
        if not ret: break
        results = model.predict(frame, conf=0.5, verbose=False)
        result = results[0]
        if result.obb and len(result.obb) > 0:
            data_stored["frame"] = frame #type: ignore
            data_stored["frame_with_boxes"] = result.plot() #type: ignore
            data_stored["coordinates"] = result.obb.cls.tolist()  # type: ignore
        sleep(0.2)

# fastAPI risponde alla chiamata GET
@app.get("/")
def get_root():
    # legge l'ultimo frame salvato
    frame = data_stored.get("frame_with_boxes")
    if frame is not None:
        # trasforma l'array num py in .jpg
        success, buffer = cv2.imencode('.jpg', frame) # type: ignore
        # codifica in bytes l'immagine per la visualizzazione corretta
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    else:
        return {"status": "Errore nel recuperare l'immagine."}
    
@app.get("/get_coordinates")
def get_coordinates():
    # legge l'ultimo frame salvato
    frame = data_stored.get("frame")
    if frame is not None:
        # trasforma l'array num py in .jpg
        success, buffer = cv2.imencode('.jpg', frame) # type: ignore
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return {"status": "Ok", "frame": f"data:image/jpeg;base64,{jpg_as_text}", "coordinates": data_stored["coordinates"]}
    else:
        return {"status": "Errore nel recuperare l'immagine."}

# lancio la funzione di cattura della camera con un thread per non bloccare il server con openCV.read
thread = threading.Thread(target=update_frame, daemon=True)
thread.start()