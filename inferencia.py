import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# Importamos las librerías EXACTAS que usa tu profesor
try:
    from picamera2 import Picamera2
    import libcamera 
    USING_PICAM = True
except ImportError:
    print("⚠️ AVISO: Picamera2 no encontrado. Usando OpenCV VideoCapture (USB).")
    USING_PICAM = False

# =========================
# 1. CONFIGURACIÓN
# =========================
DEVICE = torch.device("cpu")
MODEL_PATH = "modelo_conv.pt" 
CLASS_NAMES = ["casco", "mascarilla", "nada"]
SKIP_FRAMES = 4

# =========================
# 2. PREPROCESADO IA
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
    # La IA recibe RGB (Lo que le gusta)
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
        camera = Picamera2()
        
        # --- CONFIGURACIÓN ESTILO PROFESOR ---
        # Usamos create_still_configuration.
        # Ponemos 'main' en 640x480 para que la Raspberry no explote procesando HD.
        # Le pedimos RGB888 explícitamente para que la IA funcione.
        config = camera.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        print("📷 Cámara Pi iniciada (Configuración Still).")
    else:
        cap = cv2.VideoCapture(0)

    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0
    
    COLORS = {
        "casco": (0, 255, 0),       # Verde
        "mascarilla": (255, 255, 0), # Cian
        "nada": (0, 0, 255)         # Rojo
    }

    try:
        while True:
            # A. CAPTURA
            if USING_PICAM:
                # Capturamos del stream "main". Viene en RGB.
                frame_rgb = camera.capture_array("main")
            else:
                ret, frame_bgr_usb = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr_usb, cv2.COLOR_BGR2RGB)

            # B. INFERENCIA (La IA usa el RGB tal cual)
            if frame_count % SKIP_FRAMES == 0:
                current_class, current_conf = predict_frame(model, frame_rgb)
            
            frame_count += 1

            # C. VISUALIZACIÓN (¡AQUÍ ESTÁ LA SOLUCIÓN!)
            # Tu cara es azul en RGB si la muestras en OpenCV.
            # Convertimos a BGR para que la ventana te vea con color normal.
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # D. DIBUJAR
            color_texto = COLORS.get(current_class, (255, 255, 255))
            
            # Barra negra arriba
            cv2.rectangle(frame_display, (0, 0), (640, 50), (0, 0, 0), -1)

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
        print("Cerrando...")
        if USING_PICAM: camera.stop()
        else: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()