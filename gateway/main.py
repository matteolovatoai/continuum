import asyncio
import io
import logging
import os

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")   # usato nella dashboard HTML

app = FastAPI(title="Vision Gateway", version="1.0.0")

# ---------------------------------------------------------------------------
# Ultimo risultato valido — condiviso con il WebSocket (come nel vostro codice)
# ---------------------------------------------------------------------------
data_stored: dict = {"status": "in attesa di immagini..."}


# ---------------------------------------------------------------------------
# Schema (specchio di ml-service per documentazione Swagger)
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    box: list[float]
    yolo_class: str
    yolo_confidence: float
    mobilenet_class: str
    mobilenet_confidence: float

class DetectResponse(BaseModel):
    detections: list[Detection]
    device: str
    image_size: list[int]


# ---------------------------------------------------------------------------
# POST /detect  — riceve immagine, chiama ml-service, salva in data_stored
# ---------------------------------------------------------------------------
@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Il file deve essere un'immagine.")

    image_bytes = await file.read()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{ML_SERVICE_URL}/detect",
                files={"file": (file.filename, io.BytesIO(image_bytes), file.content_type)},
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="ML service non raggiungibile.")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="ML service timeout.")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"ML service error: {exc.response.text}")

    result = response.json()

    # Salva l'ultimo risultato valido per il WebSocket
    data_stored["last_result"] = result
    data_stored["status"] = "ok"

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# GET /health  — controlla gateway + ml-service
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    status = {}
    status["status"] = {"gateway": "ok"}
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{ML_SERVICE_URL}/health")
            status["ml_service"] = r.json()
        except Exception as exc:
            status["ml_service"] = {"status": "unreachable", "error": str(exc)}
    return status


# ---------------------------------------------------------------------------
# WebSocket /ws  — pusha l'ultimo risultato ai client connessi
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            last = data_stored.get("last_result")
            if last:
                await websocket.send_json(last)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnesso.")


# ---------------------------------------------------------------------------
# GET /dashboard  — UI minimale per vedere i risultati in live
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Vision Pipeline Dashboard</title>
    <style>
        body {{ font-family: monospace; background: #111; color: #eee; padding: 2rem; }}
        h1 {{ color: #7cf; }}
        #detections {{ background: #222; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
        .det {{ border-left: 3px solid #7cf; padding: 0.4rem 0.8rem; margin: 0.5rem 0; }}
        .label {{ color: #afc; font-weight: bold; }}
        .conf {{ color: #fa8; }}
    </style>
</head>
<body>
    <h1>Vision Pipeline — Live Results</h1>
    <p id="status">In attesa di dati...</p>
    <div id="detections"></div>

    <script>
        const ws = new WebSocket("ws://{SERVER_HOST}:8000/ws");

        ws.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            document.getElementById("status").innerText =
                "Device: " + data.device + " | Immagine: " + data.image_size.join("x");

            const div = document.getElementById("detections");
            if (!data.detections || data.detections.length === 0) {{
                div.innerHTML = "<p>Nessun oggetto rilevato.</p>";
                return;
            }}
            div.innerHTML = data.detections.map((d, i) => `
                <div class="det">
                    <b>#${{i+1}}</b>
                    YOLO: <span class="label">${{d.yolo_class}}</span>
                    (<span class="conf">${{(d.yolo_confidence*100).toFixed(1)}}%</span>)
                    &nbsp;→&nbsp;
                    MobileNet: <span class="label">${{d.mobilenet_class}}</span>
                    (<span class="conf">${{(d.mobilenet_confidence*100).toFixed(1)}}%</span>)
                    &nbsp;| box: [${{d.box.map(v=>v.toFixed(0)).join(", ")}}]
                </div>
            `).join("");
        }};

        ws.onerror = () => document.getElementById("status").innerText = "WebSocket: errore di connessione";
        ws.onclose = () => document.getElementById("status").innerText = "WebSocket: disconnesso";
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)
