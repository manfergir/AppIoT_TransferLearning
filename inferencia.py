import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

try:
    from picamera2 import Picamera2
    import libcamera 
    USING_PICAM = True
except ImportError:
    print("AVISO: Picamera2 no encontrado. Usando OpenCV VideoCapture (USB).")
    USING_PICAM = False

# 1. Configuración inicial
DEVICE = torch.device("cpu")
MODEL_PATH = "modelo_conv.pt" 
CLASS_NAMES = ["casco", "mascarilla", "nada"]
SKIP_FRAMES = 4

# 2. Preprocesado y carga del modelo
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
        print(f"ERROR: No se encuentra ningún state_dict en '{MODEL_PATH}'")
        exit()
    model.to(DEVICE)
    model.eval()
    return model

# 3. Predicción
def predict_frame(model, frame_rgb):
    pil_img = Image.fromarray(frame_rgb)
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    return CLASS_NAMES[pred_idx.item()], conf.item()

# 4. Bucle principal
def main():
    model = load_model()

    if USING_PICAM:
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        print("PiCamera iniciada con éxito.")
    else:
        cap = cv2.VideoCapture(0)

    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0
    
    # Colores (B, G, R) para el texto
    COLORS = {
        "casco": (0, 255, 0),        # Verde
        "mascarilla": (255, 255, 0), # Azul clarito
        "nada": (0, 0, 255)          # Rojo
    }

    try:
        while True:
            # A. CAPTURA
            if USING_PICAM:
                frame_bgr = camera.capture_array("main")
            else:
                ret, frame_bgr = cap.read()
                if not ret: break

            # B. INFERENCIA
            if frame_count % SKIP_FRAMES == 0:
                frame_rgb_ia = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                current_class, current_conf = predict_frame(model, frame_rgb_ia)
            
            frame_count += 1

            cv2.rectangle(frame_bgr, (0, 0), (640, 50), (0, 0, 0), -1)

            color_texto = COLORS.get(current_class, (255, 255, 255))
            text = f"DETECTADO: {current_class.upper()} ({current_conf*100:.1f}%)"
            
            cv2.putText(frame_bgr, text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)

            cv2.imshow("AppIoT: sistema de seguridad usando transfer learning", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Cerrando proyecto...")
        if USING_PICAM: camera.stop()
        else: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
