"""API de inferencia — conteo de girasoles."""

import os
import time
import tempfile
import subprocess
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


BOX_COLOR = (255, 80, 0)
BOX_W = 1
BAR_H = 40


def dibujar_cajas(frame: np.ndarray, boxes) -> np.ndarray:
    anotada = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(anotada, (x1, y1), (x2, y2), BOX_COLOR, BOX_W)
    return anotada


def dibujar_overlay_conteo(frame: np.ndarray, texto: str) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - BAR_H), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, texto, (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def inferir_frame(img: np.ndarray):
    results = model.predict(img, conf=0.4, verbose=False)
    conteo = len(results[0].boxes)
    anotada = dibujar_cajas(img, results[0].boxes)
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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp_out:
                raw_output_path = tmp_out.name
            output_path = raw_output_path.replace(".avi", ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            ids_vistos = set()
            results = model.track(
                source=temp_filename,
                conf=0.4,
                tracker="bytetrack.yaml",
                persist=True,
                stream=True,
                verbose=False,
            )
            for result in results:
                frame = dibujar_cajas(result.orig_img, result.boxes)
                if result.boxes.id is not None:
                    for tid in result.boxes.id.tolist():
                        ids_vistos.add(int(tid))
                texto = f"En pantalla: {len(result.boxes)}   |   Total vistos: {len(ids_vistos)}"
                frame = dibujar_overlay_conteo(frame, texto)
                writer.write(frame)
            writer.release()
            conteo = len(ids_vistos)

            # Convertimos a H.264 para que el navegador pueda reproducir el video
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_output_path, "-c:v", "libx264", "-crf", "23", "-preset", "fast", output_path],
                check=True, capture_output=True,
            )

            with open(output_path, "rb") as f:
                video_bytes = f.read()
        finally:
            for path in (temp_filename, raw_output_path, output_path):
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
