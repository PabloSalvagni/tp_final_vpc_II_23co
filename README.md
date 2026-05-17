# Trabajo Final de Vision por Computadora II

Repositorio de Trabajo Final de Vision por Computadora II

- Federica Pavese
- Pablo Salvagni
- Rodrigo Parra

## Título TP

Conteo automático de capítulos (flores) de girasol para asistencia en la estimación de rendimiento agrícola.

## Problema

En la agricultura de precisión, la estimación temprana del rendimiento (rinde) es vital para la logística y la comercialización. Actualmente, el conteo de capítulos (flores) de girasol se realiza de forma manual y por muestreo estadístico, lo cual es:

- **Lento y costoso:** Requiere personal recorriendo el lote, ahora mejoró con el uso de la tecnología, como drones.
- **Propenso al error humano:** La fatiga visual y la variabilidad en la densidad del cultivo afectan la precisión.
- **No escalable:** Es imposible contar manualmente cada flor en campos de cientos de hectáreas.
  **Objetivo:** Desarrollar un prototipo basado en Visión por Computadora que automatice este proceso, proporcionando un mecanismo automatizado de conteo a partir de imágenes aéreas o terrestres.

## Solución

Para resolver este problema, se propone la implementación de un modelo de **Detección de Objetos** basado en la arquitectura **YOLOv8** (You Only Look Once)
**Componentes clave:**

- **Transfer Learning:** Utilizaremos pesos pre-entrenados en el dataset COCO para acelerar la convergencia y mejorar la extracción de características, realizando un fine-tuning específico para la clase "sunflower".
- **Pipeline de Procesamiento:**

1. Redimensionamiento y normalización de imágenes.
2. Inferencia mediante el detector para localizar cada capítulo.
3. Lógica de conteo basada en la cantidad de bounding boxes detectadas con un umbral de confianza específico.
   **Métricas de Validación:** La solución se validará mediante el cálculo del **mAP (mean Average Precision)** y la **Latencia de Inferencia**, evaluando la factibilidad de uso en escenarios de tiempo real, como dispositivos móviles y drones

#### Análisis del Dataset (Roboflow)

El dataset seleccionado para realizar el [TP es Sunflower (ryan 8018)](https://universe.roboflow.com/raiyan8018/sunflower-mn2cr)

**Ficha Técnica para el Paper:**

- **Origen:** Roboflow Universe.
- **Tarea:** Object Detection (Bounding Boxes).
- **Etiquetas:** Identificación de capítulos de girasol (Sunflower heads).
- **Pre-procesamiento sugerido:** Dada la naturaleza del cultivo, aplicaremos Data Augmentation (rotaciones aleatorias y cambios de brillo/saturación) para simular diferentes condiciones de luz solar y ángulos de toma de cámara, evitando el overfitting.
- **División de datos:** Se respetará una estructura de 70% entrenamiento, 20% validación y 10% testeo para asegurar la robustez de los resultados.

---

## Guía de Desarrollo y Colaboración

Para mantener el proyecto reproducible y ordenado entre los integrantes con y sin GPU, adoptamos un flujo de trabajo híbrido.

### 1. Configuración del Entorno Local (PC con CPU/GPU local)

Para trabajar localmente (análisis EDA liviano o inferencia en tiempo real), utilizamos `uv` como gestor de paquetes ultrarrápido:

1. **Clonar el repositorio y moverte a la rama de trabajo:**

   ```bash
   git clone https://github.com/PabloSalvagni/tp_final_vpc_II_23co.git
   cd tp_final_vpc_II_23co
   git fetch origin
   git checkout branch_name/

   ```

2. **Crear y activar entorno con uv:**

   ```bash
   uv venv
   uv run uv pip install -r requirements.txt
   ```

3. **Configurar credenciales privadas:**
   - Crear archivo `.env` en la raíz del proyecto:
     ```bash
     touch .env
     ```
   - Agregar la siguiente línea:

     ```bash
     ROBOFLOW_API_KEY=tu_api_key
     ```

### 2. Configuración Colab

Como **Colab** será el entorno central de ejecución, cada vez que se inicie sesión en una notebook nueva, se debe ejecutar el siguiente bloque en la primera celda para clonar el repositorio e instalar el entorno:

```
# 1. Instalar lo que Colab no trae por defecto
!pip install ultralytics roboflow python-dotenv

# 2. Solo si van a usar módulos .py de la carpeta src (si no, borren estas 3 líneas)
import os, sys
!git clone -b branch_name https://github.com/PabloSalvagni/tp_final_vpc_II_23co.git
sys.path.append(os.path.abspath("tp_final_vpc_II_23co"))
```
