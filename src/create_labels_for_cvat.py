from pathlib import Path
from ultralytics import YOLO
import sys

# Setup dei percorsi (System Design: modularità)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils import DIR_DATA, DIR_MODELS

CVAT_IMPORT_PATH = DIR_DATA / "cvat-import"
INPUT_IMAGES_PATH = CVAT_IMPORT_PATH / "images"
model_path = DIR_MODELS / "yolo_labeling_alpha" / "weights" / "best.pt"

model = YOLO(str(model_path), task="obb")

results = model.predict(
    source=str(INPUT_IMAGES_PATH),
    project=CVAT_IMPORT_PATH,
    name="alpha_version",
    save_txt=True,
    conf=0.5
)