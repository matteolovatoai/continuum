from pathlib import Path

DIR_PROJECT = Path(__file__).resolve().parent.parent
DIR_DATA = DIR_PROJECT / "data"
DIR_DETECTION = DIR_DATA / "detection"
DIR_AI = DIR_PROJECT / "ai-service"
DIR_WEB = DIR_PROJECT / "web"
DIR_MODELS = DIR_PROJECT / "models"
DIR_YOLO_ALPHA = DIR_MODELS / "yolo_labeling_alpha" / "weights" / "best.pt"


if __name__ == "__main__":
    print(f"La home del progetto è: {DIR_PROJECT}")
    print(f"I dati sono su: {DIR_DATA}")