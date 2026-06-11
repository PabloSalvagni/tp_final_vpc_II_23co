# Conteo de Girasoles — Deploy con Docker

Stack de inferencia para detectar y contar capítulos de girasol en imágenes y videos de campo, usando un modelo **YOLOv11s** servido por FastAPI y una interfaz web en Streamlit.

---

## Arquitectura

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│  PostgreSQL  │
│ 127.0.0.1   │     │ 127.0.0.1   │     │  solo red    │
│  :8501      │     │  :8000      │     │  interna     │
└─────────────┘     └─────────────┘     └──────────────┘
```

| Servicio | Rol |
|---|---|
| **streamlit** | Interfaz web: subida de archivos, vista previa y visualización de resultados con bounding boxes |
| **api** | Inferencia con YOLO, devuelve imagen/video anotado y registra métricas en Postgres |
| **db** | Almacena cada inferencia en la tabla `registro_inferencias` |

---

## Estructura del directorio

```
deploy/
├── .env.example            # Plantilla de variables de entorno (copiar a .env)
├── docker-compose.yml      # Orquestación de los 3 servicios
├── api/
│   ├── main.py             # Endpoints /predict y /predict/frame
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit/
│   ├── app.py              # Interfaz de conteo
│   ├── Dockerfile
│   └── requirements.txt
├── pesos/
│   └── best.pt             # Pesos del modelo (no incluido en el repo — ver abajo)
└── README.md
```

---

## Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (o Docker Engine + Docker Compose plugin)
- Al menos **4 GB de RAM** disponibles para Docker
- El archivo de pesos del modelo: `best.pt`

### Configurar variables de entorno (obligatorio)

El stack **no levanta sin un archivo `.env`**. Las credenciales no están en el repositorio.

```bash
cd deploy
cp .env.example .env

# Generar contraseña para Postgres
openssl rand -base64 24
# Pegar el resultado en POSTGRES_PASSWORD dentro de .env
```

| Variable | Descripción |
|---|---|
| `POSTGRES_USER` | Usuario de Postgres |
| `POSTGRES_PASSWORD` | Contraseña (obligatoria, definir en `.env`) |
| `POSTGRES_DB` | Nombre de la base |
| `API_PORT` | Puerto local de FastAPI (default `8000`) |
| `STREAMLIT_PORT` | Puerto local de Streamlit (default `8501`) |
| `ENABLE_DOCS` | `true` solo si necesitás Swagger en `/docs`; por defecto `false` |

> La conexión API → Postgres se arma sola en `docker-compose.yml` a partir de las variables `POSTGRES_*`. No hace falta definir `DATABASE_URL` manualmente.

### Colocar el modelo

El contenedor de la API espera el modelo en `deploy/pesos/best.pt`. Si aún no existe la carpeta:

```bash
cd deploy
mkdir -p pesos
cp /ruta/a/tu/best.pt pesos/best.pt
```

> Sin este archivo la API arranca pero devuelve error 500 al inferir.

---
Para obtener el último `best.pt` entrenado por el modelo, acceder a la rama main y buscarlo en: 
```bash
/models/yolo11s_sgd-4/best.pt
```

---

## Levantar el ambiente

### 1. Ir al directorio de deploy y configurar `.env`

```bash
cd deploy
cp .env.example .env
# Completar POSTGRES_PASSWORD (ver sección anterior)
```

### 2. Construir y levantar los contenedores

```bash
docker compose up --build -d
```

La primera vez puede tardar varios minutos (descarga de imágenes base, instalación de `ultralytics`, OpenCV, etc.).

### 3. Verificar que los servicios estén corriendo

```bash
docker compose ps
```

Deberías ver tres contenedores activos: `vpc_postgres`, `vpc_api` y `vpc_streamlit`.

### 4. Revisar logs (si algo falla)

```bash
# Todos los servicios
docker compose logs -f

# Solo la API
docker compose logs -f api

# Solo Streamlit
docker compose logs -f streamlit
```

En los logs de `api` debería aparecer:

```
Modelo YOLO cargado correctamente.
Base de datos inicializada correctamente.
```

---

## Probar el conteo de girasoles

### Opción Visual — Interfaz Streamlit (UI)

1. Abrir en el navegador: **http://127.0.0.1:8501**

2. En la columna **Entrada**:
   - Subir una imagen (`.jpg`, `.png`) o un video (`.mp4`)
   - Ver la vista previa en el panel superior
   - Si es video, elegir el modo:
     - **Frame a frame**: muestra los bounding boxes mientras procesa
     - **Procesar completo**: más rápido; devuelve el video anotado al final

3. Pulsar **Procesar archivo**

4. En la columna **Resultado**:
   - **Imagen**: se muestra la foto con bounding boxes y el mensaje de conteo debajo
   - **Video (completo)**: se reproduce el MP4 anotado con el conteo acumulado
   - **Video (frame a frame)**: se actualiza frame a frame con el conteo acumulado en tiempo real

### Opción Code — API directa (curl)

Por defecto **Swagger está deshabilitado** (`ENABLE_DOCS=false`). Para habilitarlo temporalmente, setear `ENABLE_DOCS=true` en `.env` y reiniciar la API. Solo accesible en `http://127.0.0.1:8000/docs`.

#### Imagen

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/ruta/a/imagen.jpg" \
  -o resultado.jpg \
  -D headers.txt

grep -i x-conteo headers.txt
```

La respuesta es un JPEG anotado. El conteo viene en el header `X-Conteo`.

#### Video

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/ruta/a/video.mp4" \
  -o resultado.mp4 \
  -D headers.txt

grep -i x-conteo headers.txt
```

La respuesta es un MP4 anotado. El conteo acumulado (suma de detecciones por frame) viene en `X-Conteo`.

#### Frame individual (para procesamiento en vivo)

```bash
curl -X POST "http://localhost:8000/predict/frame" \
  -F "file=@/ruta/a/frame.jpg" \
  -o frame_anotado.jpg \
  -D headers.txt
```

---

## Consultar métricas en Postgres

Cada inferencia exitosa se guarda en `registro_inferencias`:

| Columna | Descripción |
|---|---|
| `id` | ID autoincremental |
| `fecha` | Timestamp de la inferencia |
| `nombre_archivo` | Nombre del archivo subido |
| `conteo` | Cantidad detectada |
| `tiempo_procesamiento_ms` | Tiempo de procesamiento en milisegundos |

Postgres **no expone puerto al host** (solo red interna Docker). Consultar desde el contenedor:

```bash
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "SELECT * FROM registro_inferencias ORDER BY fecha DESC LIMIT 10;"
```

> Reemplazá `$POSTGRES_USER` y `$POSTGRES_DB` por los valores de tu `.env`, o exportalos antes: `set -a && source .env && set +a`

---

## Servicios y puertos

Todos los servicios accesibles desde el navegador/host están ligados a **127.0.0.1** (solo tu máquina).

| Servicio | URL local | Notas |
|---|---|---|
| **Streamlit** | http://127.0.0.1:8501 | Interfaz principal |
| **FastAPI** | http://127.0.0.1:8000 | Sin Swagger por defecto |
| **PostgreSQL** | no expuesto | Solo vía `docker compose exec` |

---

## Desarrollo local

Los volúmenes en `docker-compose.yml` montan el código en vivo:

- `./api` → `/app` en el contenedor `api`
- `./streamlit` → `/app` en el contenedor `streamlit`

Si modificás `streamlit/app.py`, Streamlit detecta el cambio al recargar la página. Tras editar `api/main.py`, reiniciar la API:

```bash
docker compose restart api
```

Para reconstruir solo un servicio tras cambiar dependencias:

```bash
docker compose up --build -d api
docker compose up --build -d streamlit
```

---

## Seguridad (uso local)

El stack está configurado para **minimizar superficie de ataque** aunque solo corra en tu máquina:

| Medida | Detalle |
|---|---|
| Sin credenciales en el repo | Contraseñas solo en `.env` (gitignored); el compose falla si faltan |
| Puertos en `127.0.0.1` | Streamlit y API no escuchan en `0.0.0.0` del host |
| Postgres aislado | Sin puerto publicado; solo la API se conecta por red Docker |
| Swagger deshabilitado | `ENABLE_DOCS=false` por defecto |
| Pesos en solo lectura | Volumen `pesos/` montado como `:ro` en la API |
| Logs sin detalles internos | Errores de BD/modelo no imprimen trazas con datos sensibles |
| Archivos temporales seguros | La API no usa el nombre del archivo subido para rutas en disco |

### Qué no commitear

- `.env` (secretos)
- `pesos/*.pt` (modelo entrenado)
- Archivos subidos por usuarios durante pruebas

### Qué sí commitear

- `.env.example` (plantilla sin secretos)
- Código fuente y `docker-compose.yml`

### Si necesitás depurar la API

Setear temporalmente en `.env`:

```env
ENABLE_DOCS=true
```

Reiniciar solo la API: `docker compose up -d api`. Volver a `false` cuando termines.

---

## Detener el ambiente

```bash
# Detener contenedores (conserva datos de Postgres)
docker compose down

# Detener y borrar volúmenes (elimina métricas guardadas)
docker compose down -v
```

---

## Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `No se pudo conectar con la API` en Streamlit | La API no está levantada o aún está cargando el modelo | `docker compose logs api` y esperar a ver "Modelo YOLO cargado" |
| Error 500 al procesar | Falta `pesos/best.pt` o está corrupto | Verificar que exista `deploy/pesos/best.pt` |
| Falla el build de `api` con `libgl1-mesa-glx` | Paquete obsoleto en Debian reciente | El Dockerfile ya usa `libgl1`; hacer `docker compose build --no-cache api` |
| Puerto 8000 o 8501 ocupado | Otro proceso usa el puerto | Cambiar `API_PORT` / `STREAMLIT_PORT` en `.env` |
| `Definir POSTGRES_PASSWORD en .env` al levantar | Falta `.env` o contraseña vacía | Completar `.env` según `.env.example` |
| Video muy lento en modo frame a frame | Una petición HTTP por frame | Usar modo **Procesar completo** para videos largos |

---

## Notas sobre el conteo

- En **imágenes**, el conteo es la cantidad de detecciones en ese frame (`conf >= 0.4`).
- En **videos**, el conteo reportado es la **suma de detecciones en todos los frames**. Si el mismo girasol aparece en varios frames, se cuenta más de una vez.
