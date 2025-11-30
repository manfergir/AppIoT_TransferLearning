import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# Importamos Picamera2 y libcamera para la configuración del profesor
try:
    from picamera2 import Picamera2
    import libcamera # Necesario para el Transform que usa tu profesor
    USING_PICAM = True
except ImportError:
    print("⚠️ AVISO: Picamera2 no encontrado. Usando OpenCV VideoCapture (USB).")
    USING_PICAM = False

# =========================
# CONFIGURACIÓN
# =========================
DEVICE = torch.device("cpu")
MODEL_PATH = "modelo_conv.pt" # Tu modelo
CLASS_NAMES = ["casco", "mascarilla", "nada"]
SKIP_FRAMES = 4 

# =========================
# PREPROCESADO
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
# CARGAR MODELO
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
# PREDICCIÓN
# =========================
def predict_frame(model, frame_rgb):
    # La IA recibe RGB (que es lo que le gusta)
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
        
        # --- CONFIGURACIÓN ESTILO PROFESOR ---
        config = picam2.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}, # Alta resolución latente
            lores={"size": (640, 480), "format": "RGB888"},  # Baja resolución para procesar (Fluidez)
            display="lores" # Usamos la pequeña para ver
        )
        
        # Aplicamos el Transform (Flip) si lo tenías en el ejemplo, 
        # aunque comentaste que el flip no lo querías, lo dejo comentado por si acaso.
        # config['transform'] = libcamera.Transform(vflip=True) 
        
        picam2.configure(config)
        picam2.start()
        print("📷 Cámara Pi iniciada (Configuración Main/Lores).")
    else:
        cap = cv2.VideoCapture(0)

    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0
    
    # Colores semánticos para el RECUADRO (Texto siempre blanco)
    COLORS = {
        "casco": (0, 255, 0),       # Verde
        "mascarilla": (255, 255, 0), # Cian
        "nada": (0, 0, 255)         # Rojo
    }
    COLOR_TEXTO = (255, 255, 255) # Blanco

    try:
        while True:
            # A. CAPTURA
            if USING_PICAM:
                # IMPORTANTE: Capturamos del stream "lores" para que vaya rápido
                # La cámara nos da RGB888 (Rojo-Verde-Azul)
                frame_rgb = picam2.capture_array("lores")
            else:
                ret, frame_bgr_usb = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr_usb, cv2.COLOR_BGR2RGB)

            # B. INFERENCIA (Usamos RGB, que es lo correcto para la IA)
            if frame_count % SKIP_FRAMES == 0:
                current_class, current_conf = predict_frame(model, frame_rgb)
            
            frame_count += 1

            # C. VISUALIZACIÓN (SOLUCIÓN DEFINITIVA DE COLOR)
            # OpenCV (cv2.imshow) espera BGR (Azul-Verde-Rojo).
            # Nosotros tenemos RGB. Si no hacemos esto, se ve azul.
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # D. DIBUJAR
            color_box = COLORS.get(current_class, (255, 255, 255))
            
            # Panel oscuro
            overlay = frame_display.copy()
            cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
            frame_display = cv2.addWeighted(overlay, 0.5, frame_display, 0.5, 0)

            text = f"DETECTADO: {current_class.upper()} ({current_conf*100:.1f}%)"
            cv2.putText(frame_display, text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXTO, 2)
            
            # Recuadro opcional alrededor
            cv2.rectangle(frame_display, (0,0), (640, 480), color_box, 4)

            cv2.imshow("Sistema TFM", frame_display)

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