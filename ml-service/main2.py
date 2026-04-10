import io
import logging
import urllib.request
import json
from contextlib import asynccontextmanager

import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Variabile globale dei modelli (caricati una volta sola al boot)
# ---------------------------------------------------------------------------
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carico YOLO...")
    models["yolo"] = YOLO("/app/models/yolo.pt")
    models["yolo"].to(DEVICE)

    logger.info("Carico MobileNetV3 (Large)...")
    mobilenet = tv_models.mobilenet_v3_large(weights=None)
    
    # Ricostruisco l'ultimo livello con 4 classi come fatto nel training
    mobilenet.classifier[3] = torch.nn.Linear(mobilenet.classifier[3].in_features, 4) #type:ignore

    # Carica lo state_dict custom bypassando i pesi di default
    state_dict = torch.load("/app/models/mobilenet_lego.pth", map_location=DEVICE)
    mobilenet.load_state_dict(state_dict)
    
    mobilenet.eval().to(DEVICE)
    models["mobilenet"] = mobilenet

    logger.info("Configuro le labels per i Lego...")
    # ImageFolder mappa automaticamente le classi in ordine alfabetico in base al nome delle cartelle
    models["labels"] = ["bianco", "blu", "giallo", "rosso"]

    logger.info(f"Modelli pronti su {DEVICE}.")
    yield
    models.clear()


app = FastAPI(title="ML Service", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# MobileNet transform (uguale per ogni crop)
# ---------------------------------------------------------------------------
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def classify_crop(crop: Image.Image) -> tuple[str, float]:
    """Classifica un singolo crop PIL con MobileNet."""
    tensor = _transform(crop).unsqueeze(0).to(DEVICE) # type: ignore
    with torch.no_grad():
        logits = models["mobilenet"](tensor)
    probs = torch.softmax(logits, dim=1)
    conf, idx = probs[0].max(dim=0)
    label = models["labels"][idx.item()]
    return label, round(conf.item(), 4)


# ---------------------------------------------------------------------------
# Schema risposta
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    box: list[float]               # [x1, y1, x2, y2] pixel
    yolo_class: str                # classe rilevata da YOLO
    yolo_confidence: float         # confidence YOLO
    mobilenet_class: str           # classe crop da MobileNet
    mobilenet_confidence: float    # confidence MobileNet


class DetectResponse(BaseModel):
    detections: list[Detection]
    device: str
    image_size: list[int]          # [width, height]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Il file deve essere un'immagine.")

    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = image.size

    # --- YOLO: rilevamento box ---
    yolo_results = models["yolo"](image, device=DEVICE, verbose=False)[0]

    detections: list[Detection] = []

    for box in yolo_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        yolo_class = yolo_results.names[int(box.cls[0].item())]
        yolo_conf = round(float(box.conf[0].item()), 4)

        # Clamp coordinate ai bordi immagine
        x1c = max(0, int(x1))
        y1c = max(0, int(y1))
        x2c = min(w, int(x2))
        y2c = min(h, int(y2))

        if x2c <= x1c or y2c <= y1c:
            continue  # box degenere, salta

        # --- MobileNet: classifica il crop ---
        crop = image.crop((x1c, y1c, x2c, y2c))
        mn_class, mn_conf = classify_crop(crop)

        detections.append(Detection(
            box=[x1, y1, x2, y2],
            yolo_class=yolo_class,
            yolo_confidence=yolo_conf,
            mobilenet_class=mn_class,
            mobilenet_confidence=mn_conf,
        ))

    return DetectResponse(
        detections=detections,
        device=DEVICE,
        image_size=[w, h],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": list(models.keys()),
    }
