import io
import os
import tempfile

import cv2
import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="TP VPC II - Conteo de Girasoles",
    page_icon="🌻",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
PREDICT_FRAME_ENDPOINT = f"{API_URL}/predict/frame"

st.title("Detector y Contador de Girasoles")
st.markdown("""
Esta herramienta utiliza un modelo **YOLOv11s SGD** entrenado específicamente para detectar capítulos de girasol en imágenes y video tomadas a campo.  
Sube una foto o video y el sistema registrará la métrica automáticamente.
""")

st.markdown(
    """
    <style>
        div.st-key-panel-entrada {
            height: 50vh;
            min-height: 280px;
            max-height: 50vh;
            overflow-y: auto;
            overflow-x: hidden;
        }
        div.st-key-panel-resultado {
            height: auto;
            min-height: 280px;
            overflow: visible;
        }
        div.st-key-panel-entrada [data-testid="stImage"],
        div.st-key-panel-entrada [data-testid="stVideo"] {
            max-height: 46vh;
            overflow: hidden;
        }
        div.st-key-panel-entrada img {
            max-height: 46vh !important;
            width: 100% !important;
            object-fit: contain !important;
        }
        div.st-key-panel-resultado img,
        div.st-key-panel-resultado video {
            width: 100% !important;
            object-fit: contain !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def mostrar_preview_video(uploaded_file):
    """Muestra el primer frame como imagen para que videos verticales no desborden el panel."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getvalue())
        video_path = tmp.name

    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo leer el video.")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Vista previa del video (primer frame)", use_container_width=True)
    finally:
        cap.release()
        os.remove(video_path)


def procesar_imagen(uploaded_file, panel):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(PREDICT_ENDPOINT, files=files, timeout=120)

    with panel:
        if response.status_code != 200:
            st.error(f"Error en el servidor: {response.status_code}")
            return

        conteo = int(response.headers.get("X-Conteo", 0))
        img_result = Image.open(io.BytesIO(response.content))

        st.image(img_result, caption=f"Resultado — {conteo} girasoles detectados", use_container_width=True)
        st.success(f"¡Análisis completado! Se detectaron **{conteo} girasoles**.")


def procesar_video_en_vivo(uploaded_file, panel):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getvalue())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    with panel:
        frame_slot = st.empty()
        metric_slot = st.empty()
        progress = st.progress(0, text="Procesando video frame a frame...")

    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            response = requests.post(PREDICT_FRAME_ENDPOINT, files=files, timeout=60)

            with panel:
                if response.status_code != 200:
                    st.error(f"Error procesando frame {frame_idx}: {response.status_code}")
                    break

                conteo_frame = int(response.headers.get("X-Conteo", 0))

                img_frame = Image.open(io.BytesIO(response.content))
                frame_slot.image(
                    img_frame,
                    caption=f"Frame {frame_idx + 1}/{total_frames}",
                    use_container_width=True,
                )
                metric_slot.metric("Girasoles en pantalla", conteo_frame)
                progress.progress(
                    min((frame_idx + 1) / total_frames, 1.0),
                    text=f"Frame {frame_idx + 1}/{total_frames}",
                )

            frame_idx += 1
    finally:
        cap.release()
        os.remove(video_path)

    with panel:
        progress.empty()
        st.success("¡Video procesado!")


def procesar_video_completo(uploaded_file, panel):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

    with panel:
        with st.spinner("Procesando video completo en la API..."):
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=600)

        if response.status_code != 200:
            st.error(f"Error en el servidor: {response.status_code}")
            return

        conteo = int(response.headers.get("X-Conteo", 0))
        st.success(f"¡Análisis completado! Conteo acumulado: **{conteo} detecciones**.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(response.content)
            output_path = tmp.name

        st.video(output_path)
        os.remove(output_path)


col_entrada, col_resultado = st.columns(2)

with col_entrada:
    st.subheader("Entrada")
    st.caption("La vista previa del archivo subido.")

    uploaded_file = st.session_state.get("file_upload")

    with st.container(key="panel-entrada", border=True):
        if uploaded_file is not None:
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption="Imagen original", use_container_width=True)
            else:
                mostrar_preview_video(uploaded_file)

    uploaded_file = st.file_uploader(
        "Subir archivo",
        type=["jpg", "png", "mp4"],
        label_visibility="collapsed",
        key="file_upload",
    )

    st.caption("Subir archivo para analizar...")
    modo_video = "en_vivo"
    procesar = False

    if uploaded_file is not None:
        if uploaded_file.type.startswith("video"):
            modo_video = st.radio(
                "Modo de visualización del video",
                options=["en_vivo", "completo"],
                format_func=lambda x: (
                    "Frame a frame (muestra boxes mientras procesa)"
                    if x == "en_vivo"
                    else "Procesar completo (más rápido, video anotado al final)"
                ),
                horizontal=True,
            )

        procesar = st.button("Procesar archivo", type="primary", use_container_width=True)

with col_resultado:
    st.subheader("Resultado")
    st.caption("Resultado de la predicción.")

    with st.container(key="panel-resultado", border=True):
        panel_resultado = st.container()

        if uploaded_file is None:
            with panel_resultado:
                st.info("Subí una imagen o video para ver el resultado acá.")
        elif not procesar:
            with panel_resultado:
                st.info("Presioná **Procesar archivo** para iniciar el análisis.")
        else:
            try:
                if uploaded_file.type.startswith("image"):
                    with st.spinner("Analizando imagen con YOLO11s SGD..."):
                        procesar_imagen(uploaded_file, panel_resultado)
                elif modo_video == "en_vivo":
                    procesar_video_en_vivo(uploaded_file, panel_resultado)
                else:
                    procesar_video_completo(uploaded_file, panel_resultado)
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API. ¿Están corriendo los contenedores de Docker?")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
