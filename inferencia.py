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
# CONFIGURACIÓN
# =========================
DEVICE = torch.device("cpu")
MODEL_PATH = "modelo_conv.pt"
CLASS_NAMES = ["casco", "mascarilla", "nada"]
SKIP_FRAMES = 4

# =========================
# PREPROCESADO IA
# =========================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# CARGA MODELO
# =========================
def load_model():
    print(f"Cargando ResNet50...")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
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
# PREDICCIÓN
# =========================
def predict_frame(model, frame_rgb):
    # OJO: Aquí la función espera recibir RGB
    pil_img = Image.fromarray(frame_rgb)
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    return CLASS_NAMES[pred_idx.item()], conf.item()

# =========================
# MAIN
# =========================
def main():
    model = load_model()

    if USING_PICAM:
        picam2 = Picamera2()
        # --- CAMBIO IMPORTANTE: CONFIGURAMOS LA CÁMARA EN BGR ---
        # Ahora la cámara habla el mismo idioma que OpenCV
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        print("📷 Cámara Pi iniciada en modo BGR.")
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
            # 1. CAPTURA (Ahora recibimos BGR directo)
            if USING_PICAM:
                frame_bgr = picam2.capture_array() 
                # ¡YA NO HACEMOS CONVERSIÓN AQUÍ! La imagen ya viene lista para mostrar.
            else:
                ret, frame_bgr = cap.read()
                if not ret: break

            # 2. INFERENCIA
            if frame_count % SKIP_FRAMES == 0:
                # La IA sigue necesitando RGB, así que invertimos SOLO para ella
                # Esto no afecta a lo que ves en pantalla
                frame_rgb_for_ai = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                current_class, current_conf = predict_frame(model, frame_rgb_for_ai)
            
            frame_count += 1

            # 3. MOSTRAR (Usamos el frame directo de la cámara)
            color = COLORS.get(current_class, (255, 255, 255))
            
            # Panel
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
            frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)

            text = f"DETECTADO: {current_class.upper()} ({current_conf*100:.1f}%)"
            cv2.putText(frame_bgr, text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Sistema de Seguridad TFM", frame_bgr)

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