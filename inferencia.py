import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

try:
    from picamera2 import Picamera2
    USING_PICAM = True
except ImportError:
    print("⚠️ AVISO: Picamera2 no encontrado. Usando OpenCV VideoCapture (USB).")
    USING_PICAM = False

# =========================
# 1. CONFIGURACIÓN
# =========================
DEVICE = torch.device("cpu")
MODEL_PATH = "modelo_conv.pt" # Asegúrate de que el nombre es correcto
CLASS_NAMES = ["casco", "mascarilla", "nada"]
SKIP_FRAMES = 4

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
# 3. CARGAR MODELO
# =========================
def load_model():
    print(f"Cargando ResNet50...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"❌ ERROR: No encuentro '{MODEL_PATH}'")
        exit()
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# 4. PREDICCIÓN
# =========================
def predict_frame(model, frame_rgb):
    pil_img = Image.fromarray(frame_rgb)
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    return CLASS_NAMES[pred_idx.item()], conf.item()

# =========================
# 5. MAIN
# =========================
def main():
    model = load_model()

    if USING_PICAM:
        picam2 = Picamera2()
        # VOLVEMOS A RGB888 (Estándar). Haremos la conversión manual luego.
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("📷 Cámara Pi iniciada.")
    else:
        cap = cv2.VideoCapture(0)

    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0
    
    # COLORES DEL TEXTO (BGR: Azul, Verde, Rojo)
    COLORS = {
        "casco": (0, 255, 0),       # Verde
        "mascarilla": (255, 255, 0), # Cian/Azul Claro
        "nada": (0, 0, 255)         # Rojo
    }

    try:
        while True:
            # A. CAPTURA (Obtenemos RGB)
            if USING_PICAM:
                frame_rgb = picam2.capture_array()
            else:
                ret, frame_bgr_usb = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr_usb, cv2.COLOR_BGR2RGB)

            # B. INFERENCIA (La IA usa el RGB tal cual)
            if frame_count % SKIP_FRAMES == 0:
                current_class, current_conf = predict_frame(model, frame_rgb)
            
            frame_count += 1

            # C. PREPARAR PANTALLA
            # --- SOLUCIÓN COLORES ---
            # Convertimos explícitamente RGB -> BGR para que la piel se vea bien
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # D. DIBUJAR (Sobre la imagen con colores corregidos)
            
            # 1. Barra negra sólida arriba (sin filtros extraños)
            cv2.rectangle(frame_display, (0, 0), (640, 50), (0, 0, 0), -1)

            # 2. Elegimos el color del texto
            color_texto = COLORS.get(current_class, (255, 255, 255))

            # 3. Escribimos
            text = f"DETECTADO: {current_class.upper()} ({current_conf*100:.1f}%)"
            cv2.putText(frame_display, text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)

            # E. MOSTRAR
            cv2.imshow("Sistema seguridad", frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if USING_PICAM: picam2.stop()
        else: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()