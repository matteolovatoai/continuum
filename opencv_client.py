"""
opencv_client.py  —  file locale, NON va nel Docker.

Cattura frame dalla camera con OpenCV (come il codice originale),
manda ogni frame al gateway via HTTP POST /detect,
e stampa i risultati. Opzionalmente mostra il frame con i box disegnati.
"""

import time
import cv2
import requests
import numpy as np

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
GATEWAY_URL = "http://localhost:8000/detect"
CAMERA_INDEX = 0          # 0 = prima webcam
INTERVAL_S = 0.2          # secondi tra un frame e l'altro (come nell'originale)
SHOW_WINDOW = True         # True = mostra finestra OpenCV con i box disegnati
CONF_THRESHOLD = 0.5       # filtro confidence lato client (opzionale)

# Colore box (BGR)
COLOR_BOX = (0, 255, 120)
COLOR_TEXT = (255, 255, 255)

# ---------------------------------------------------------------------------
# Funzione per disegnare i box sul frame
# ---------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        yolo_cls = det["yolo_class"]
        mn_cls = det["mobilenet_class"]
        mn_conf = det["mobilenet_confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

        label = f"{yolo_cls} | {mn_cls} {mn_conf*100:.0f}%"
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 9, y1), COLOR_BOX, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Errore: impossibile aprire la camera.")
        return

    print(f"Camera aperta. Invio frame a {GATEWAY_URL} ogni {INTERVAL_S}s")
    print("Premi Q nella finestra OpenCV (o Ctrl+C) per uscire.\n")

    session = requests.Session()   # riuso connessione TCP

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Errore lettura frame, esco.")
            break

        # Codifica il frame come JPEG in memoria
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        # Manda al gateway
        try:
            response = session.post(
                GATEWAY_URL,
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            detections = result.get("detections", [])
            print(f"[{result.get('device','?')}] {len(detections)} oggetti rilevati:")
            for i, det in enumerate(detections, 1):
                print(f"  #{i} YOLO={det['yolo_class']} ({det['yolo_confidence']*100:.1f}%)"
                      f" → MobileNet={det['mobilenet_class']} ({det['mobilenet_confidence']*100:.1f}%)"
                      f" | box={[round(v) for v in det['box']]}")

            # Disegna i box sul frame e mostra
            if SHOW_WINDOW:
                annotated = draw_detections(frame.copy(), detections)
                cv2.imshow("Vision Pipeline", annotated)

        except requests.exceptions.ConnectionError:
            print("Gateway non raggiungibile, riprovo...")
        except requests.exceptions.Timeout:
            print("Timeout risposta gateway.")
        except Exception as exc:
            print(f"Errore: {exc}")

        # Uscita con tasto Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(INTERVAL_S)

    camera.release()
    cv2.destroyAllWindows()
    print("Chiuso.")


if __name__ == "__main__":
    main()
