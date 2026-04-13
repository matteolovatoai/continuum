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
REQUEST_TIMEOUT = 60       # secondi — CPU è lenta, serve margine

# Colore box (BGR)
COLOR_BOX = (0, 255, 120)
COLOR_TEXT = (0, 0, 0)

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
        cv2.rectangle(frame, (x1, y1 - 55), (x1 + len(label) * 26, y1), COLOR_BOX, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLOR_TEXT, 3)
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Errore: impossibile aprire la camera.")
        return

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manuale, 3 = automatico
    print("Autofocus e auto-esposizione disattivati (supporto dipende dalla camera).")

    print(f"Camera aperta. Invio frame a {GATEWAY_URL} ogni {INTERVAL_S}s")
    print("Premi Q nella finestra OpenCV (o Ctrl+C) per uscire.\n")

    session = requests.Session()   # riuso connessione TCP

    # Warmup: prima inference su CPU è sempre lenta, falla prima del loop
    print("Warmup in corso (prima inference su CPU è lenta)...")
    try:
        warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", warmup_frame)
        session.post(GATEWAY_URL, files={"file": ("warmup.jpg", buf.tobytes(), "image/jpeg")}, timeout=REQUEST_TIMEOUT)
        print("Warmup completato.\n")
    except Exception:
        print("Warmup fallito (ignorato), continuo.\n")

    last_annotated = None   # ultimo frame con i box disegnati

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Errore lettura frame, esco.")
            break

        # Mostra l'ultimo annotato sovrapposto al frame live (finestra unica)
        display = frame.copy()
        if last_annotated is not None:
            cv2.imshow("Vision Pipeline | A=scatta  Q=esci", last_annotated)
        else:
            cv2.imshow("Vision Pipeline | A=scatta  Q=esci", display)

        key = cv2.waitKey(1)
        # Uscita con tasto Q
        if key == ord("q"):
            break
        elif key == ord("a"):
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            # Manda al gateway
            try:
                print("Invio in corso...")
                response = session.post(
                    GATEWAY_URL,
                    files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()

                detections = result.get("detections", [])
                print(f"[{result.get('device','?')}] {len(detections)} oggetti rilevati:")
                for i, det in enumerate(detections, 1):
                    print(f"  #{i} YOLO={det['yolo_class']} ({det['yolo_confidence']*100:.1f}%)"
                        f" → MobileNet={det['mobilenet_class']} ({det['mobilenet_confidence']*100:.1f}%)"
                        f" | box={[round(v) for v in det['box']]}")

                # Disegna i box e salva come ultimo annotato — rimane visibile
                last_annotated = draw_detections(frame.copy(), detections)
                cv2.imshow("Vision Pipeline | A=scatta  Q=esci", last_annotated)

            except requests.exceptions.ConnectionError:
                print("Gateway non raggiungibile, riprovo...")
            except requests.exceptions.Timeout:
                print("Timeout risposta gateway.")
            except Exception as exc:
                print(f"Errore: {exc}")


    camera.release()
    cv2.destroyAllWindows()
    print("Chiuso.")


if __name__ == "__main__":
    main()
