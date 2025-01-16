import cv2
import os
import zipfile
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from fastapi.responses import JSONResponse

# Agregar CORS a FastAPI
app = FastAPI()

origins = [
    "http://localhost:4200",  # URL de tu frontend Angular
    "http://localhost:8100",   # URL de tu frontend Angular ionic
    "http://192.168.0.2:8100"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo YOLO
model_path = 'ultralyticsplus/yolov8s.pt'  # Ruta al archivo del modelo
model = YOLO(model_path)

# Configurar parámetros del modelo
model.conf = 0.25
model.iou = 0.45
model.max_det = 1000

# Clases del modelo
class_map = [
    'persona', 'bicicleta', 'coche', 'moto',
    'avión', 'autobús', 'tren', 'camión', 'barco',
    'semáforo', 'boca de incendio', 'señal de alto',
    'parquímetro', 'banco', 'pájaro', 'gato', 'perro',
    'caballo', 'oveja', 'vaca', 'elefante', 'oso',
    'cebra', 'jirafa', 'mochila', 'paraguas', 'bolso',
    'corbata', 'maleta', 'frisbee', 'esquís', 'snowboard',
    'pelota deportiva', 'cometa', 'bate de béisbol',
    'guante de béisbol', 'patín', 'tabla de surf',
    'raqueta de tenis', 'botella', 'copa de vino',
    'copa', 'tenedor', 'cuchillo', 'cuchara',
    'cuenco', 'plátano', 'manzana', 'sándwich',
    'naranja', 'brócoli', 'zanahoria',
    'perrito caliente', 'pizza', 'dona', 'pastel',
    'silla', 'sofá', 'planta en maceta', 'cama',
    'mesa de comedor', 'inodoro', 'televisión',
    'computadora portátil', 'ratón', 'control remoto',
    'teclado', 'teléfono móvil', 'microondas', 'horno',
    'tostadora', 'fregadero', 'nevera', 'libro', 'reloj',
    'jarrón', 'tijeras', 'oso de peluche', 'secador de pelo',
    'cepillo de dientes'
]

@app.post("/upload/")
async def upload_video(
    video: UploadFile = File(...),
    class_name: str = Form(...),
):
    if class_name not in class_map:
        return JSONResponse(status_code=400, content={"message": f"Clase no válida: {class_name}"})

    class_index = class_map.index(class_name)

    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        detections = []

        # Crear un directorio temporal para las imágenes
        output_dir = 'detections'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar la inferencia con YOLO
            results = model(frame)

            # Dibujar los cuadros alrededor de los objetos detectados
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == class_index:
                        # Dibujar el rectángulo (cuadro delimitador) alrededor del objeto
                        x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                        # Añadir texto con el nombre de la clase
                        label = class_map[class_index]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        color = (0, 255, 0)  # Color del texto
                        thickness = 2
                        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), font, font_scale, color, thickness)

                        frame_number = len(detections) + 1
                        detections.append({
                            "frame": frame_number,
                            "coordinates": box.xyxy.tolist(),
                            "confidence": float(box.conf),
                        })

                        # Guardar la imagen detectada en el directorio
                        img_filename = f"{output_dir}/frame_{frame_number}.jpg"
                        cv2.imwrite(img_filename, frame)

        cap.release()
        os.remove(temp_video_path)

        # Crear el archivo ZIP con las imágenes
        zip_filename = "detections.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

        # Eliminar las imágenes después de crear el ZIP
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(output_dir)

        # Devolver el archivo ZIP como respuesta
        return FileResponse(zip_filename, media_type='application/zip', filename=zip_filename)

    except Exception as e:
        os.remove(temp_video_path)
        return JSONResponse(status_code=500, content={"message": f"Error al procesar el video: {str(e)}"})
