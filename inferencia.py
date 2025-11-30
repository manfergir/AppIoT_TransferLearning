import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# Intenta importar Picamera2 (Raspberry Pi OS Bullseye/Bookworm)
try:
    from picamera2 import Picamera2
    USING_PICAM = True
except ImportError:
    print("⚠️ AVISO: Picamera2 no encontrado. Usando OpenCV VideoCapture (USB).")
    USING_PICAM = False

# =========================
# 1. CONFIGURACIÓN
# =========================

DEVICE = torch.device("cpu") # La Pi usa CPU
MODEL_PATH = "modelo_casco_mascarilla_tfm.pt" # <--- TU ARCHIVO .pt

# ¡OJO! Orden ALFABÉTICO estricto de tus carpetas de entrenamiento
CLASS_NAMES = ["casco", "mascarilla", "nada"]

# Configuración de fluidez
SKIP_FRAMES = 4  # Hacer inferencia solo cada X frames para que el vídeo no se trabe

# =========================
# 2. PREPROCESADO
# =========================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# 3. CARGAR EL MODELO
# =========================
def load_model():
    print(f"Cargando arquitectura ResNet50...")
    # weights=None: No descargamos nada de internet
    model = models.resnet50(weights=None)

    # Reconstruimos la capa final IGUAL que en el entrenamiento
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )

    print(f"Cargando pesos desde {MODEL_PATH}...")
    try:
        # map_location='cpu' es VITAL para pasar de PC (GPU) a Pi (CPU)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"❌ ERROR: No encuentro el archivo '{MODEL_PATH}'.")
        print("Asegúrate de copiarlo a esta misma carpeta.")
        exit()

    model.to(DEVICE)
    model.eval() # Congelar Dropout y BatchNorm
    print("✅ Modelo cargado y listo.")
    return model

# =========================
# 4. FUNCIÓN DE PREDICCIÓN
# =========================
def predict_frame(model, frame_rgb):
    # Convertir array numpy a imagen PIL
    pil_img = Image.fromarray(frame_rgb)
    
    # Transformar
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(x)
        # Softmax para obtener porcentajes
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        
    return CLASS_NAMES[pred_idx.item()], conf.item()

# =========================
# 5. BUCLE PRINCIPAL
# =========================
def main():
    # Cargar modelo
    model = load_model()

    # Iniciar Cámara
    if USING_PICAM:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("📷 Cámara Pi iniciada.")
    else:
        cap = cv2.VideoCapture(0)
        print("📷 Cámara USB iniciada.")

    # Variables de estado
    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0
    
    # Colores (BGR para OpenCV)
    COLORS = {
        "casco": (0, 255, 0),      # Verde
        "mascarilla": (255, 255, 0), # Cian/Azulito
        "nada": (0, 0, 255)        # Rojo
    }

    try:
        while True:
            # A. CAPTURA
            if USING_PICAM:
                frame_rgb = picam2.capture_array()
                # Para mostrar en OpenCV necesitamos BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame_bgr = cap.read()
                if not ret: break
                # Para la IA necesitamos RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # B. INFERENCIA (Solo cada X frames para ir fluido)
            if frame_count % SKIP_FRAMES == 0:
                # Llamamos a la IA
                current_class, current_conf = predict_frame(model, frame_rgb)
            
            frame_count += 1

            # C. DIBUJAR RESULTADOS
            # Seleccionar color según la predicción actual
            color = COLORS.get(current_class, (255, 255, 255))
            
            # Crear panel superior semitransparente para el texto
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
            frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)

            # Texto de predicción
            text = f"DETECTADO: {current_class.upper()} ({current_conf*100:.1f}%)"
            cv2.putText(frame_bgr, text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # D. MOSTRAR VENTANA
            cv2.imshow("Sistema de Seguridad TFM", frame_bgr)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Cerrando sistema...")
        if USING_PICAM:
            picam2.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()