"""API de inferencia — conteo de girasoles."""

import os
import time
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO
from sqlalchemy import create_engine, text


_enable_docs = os.getenv("ENABLE_DOCS", "false").lower() == "true"
app = FastAPI(
    title="API de Conteo de Girasoles",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# Defino extenciones de video y imagen soportadas.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def init_db():
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS registro_inferencias (
                    id SERIAL PRIMARY KEY,
                    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    nombre_archivo VARCHAR(255),
                    conteo INTEGER,
                    tiempo_procesamiento_ms REAL
                )
            """))
            conn.commit()
        print("Base de datos inicializada correctamente.")
    except Exception:
        print("Error conectando a la base de datos. Revisar DATABASE_URL y que el servicio db esté activo.")


def guardar_metrica(nombre: str, conteo: int, tiempo_ms: float):
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO registro_inferencias (nombre_archivo, conteo, tiempo_procesamiento_ms)
                VALUES (:nombre, :conteo, :tiempo)
            """), {"nombre": nombre, "conteo": conteo, "tiempo": tiempo_ms})
            conn.commit()
    except Exception:
        print("Advertencia: No se pudo guardar la métrica en Postgres.")


def leer_imagen(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")
    return img


def inferir_frame(img: np.ndarray):
    results = model.predict(img, conf=0.4, verbose=False)
    conteo = len(results[0].boxes)
    anotada = results[0].plot()
    return conteo, anotada


def codificar_jpeg(img: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise HTTPException(status_code=500, detail="Error al codificar la imagen procesada.")
    return encoded.tobytes()


MODEL_PATH = "pesos/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print("Modelo YOLO cargado correctamente.")
except Exception:
    print("No se pudo cargar el modelo. Verificar que exista pesos/best.pt.")
    model = None

init_db()


@app.post("/predict/frame")
async def predict_frame(file: UploadFile = File(...)):
    """Inferencia sobre un único frame (imagen). Devuelve JPEG anotado y conteo en header."""
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo YOLO no está disponible.")

    contents = await file.read()
    img = leer_imagen(contents)
    conteo, anotada = inferir_frame(img)

    return Response(
        content=codificar_jpeg(anotada),
        media_type="image/jpeg",
        headers={"X-Conteo": str(conteo)},
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo YOLO no está disponible.")

    start_time = time.time()
    ext = os.path.splitext(file.filename or "")[1].lower()
    contents = await file.read()

    if ext in VIDEO_EXTENSIONS:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            temp_filename = tmp.name

        try:
            cap = cv2.VideoCapture(temp_filename)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="No se pudo abrir el video.")

            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                output_path = tmp_out.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            conteo = 0
            results = model.predict(source=temp_filename, conf=0.4, stream=True, verbose=False)
            for result in results:
                conteo += len(result.boxes)
                writer.write(result.plot())
            writer.release()

            with open(output_path, "rb") as f:
                video_bytes = f.read()
        finally:
            for path in (temp_filename, "temp_output.mp4"):
                if os.path.exists(path):
                    os.remove(path)

        tiempo_ms = (time.time() - start_time) * 1000
        guardar_metrica(file.filename, conteo, tiempo_ms)

        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={"X-Conteo": str(conteo)},
        )

    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato no soportado. Usá jpg, png o mp4.")

    img = leer_imagen(contents)
    conteo, anotada = inferir_frame(img)
    tiempo_ms = (time.time() - start_time) * 1000
    guardar_metrica(file.filename, conteo, tiempo_ms)

    return Response(
        content=codificar_jpeg(anotada),
        media_type="image/jpeg",
        headers={"X-Conteo": str(conteo)},
    )
