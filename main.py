from time import sleep
import threading
from fastapi import FastAPI

app = FastAPI()

data_stored = {"count": 0}

def update_count():
    while True:
        data_stored["count"] = data_stored["count"] + 1
        sleep(2)

@app.get("/")
def get_root():
    return {"status": "OK"}

thread = threading.Thread(target=update_count, daemon=True)
thread.start()