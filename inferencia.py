import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# Intento usar exactamente las librerías que usa tu profesor
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
WINDOW_NAME = "Sistema seguridad"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

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
    print("Cargando ResNet50...")
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
    # frame_rgb: numpy array en formato RGB, dtype=uint8
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
        # <-- Cambio importante: pedimos BGR888 explícito para evitar
        # problemas de intercambio de canales. Así la imagen que devuelve
        # capture_array() ya está en BGR y se muestra correctamente con OpenCV.
        config = camera.create_still_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
            lores={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
        # Opcional: aplicar transformaciones que use tu profesor (vflip, etc.)
        # config['transform'] = libcamera.Transform(vflip=True)
        camera.configure(config)
        camera.start()
        print("📷 Cámara Pi iniciada (Formato BGR888).")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0

    COLORS = {
        "casco": (0, 255, 0),        # BGR para dibujar (verde)
        "mascarilla": (0, 255, 255), # BGR (amarillo)
        "nada": (0, 0, 255)          # BGR (rojo)
    }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            # A. CAPTURA
            if USING_PICAM:
                # Con format="BGR888" capture_array devuelve imagen en BGR
                frame_bgr = camera.capture_array("main")
                if frame_bgr is None:
                    print("Falló captura Picamera2.")
                    break
                frame_display = frame_bgr  # ya en BGR para mostrar con OpenCV
                # Convertimos a RGB para la red
                frame_rgb_for_model = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            else:
                ret, frame_bgr_usb = cap.read()
                if not ret:
                    print("Falló captura USB.")
                    break
                frame_display = frame_bgr_usb  # OpenCV ya devuelve BGR
                frame_rgb_for_model = cv2.cvtColor(frame_bgr_usb, cv2.COLOR_BGR2RGB)

            # B. INFERENCIA (la IA usa RGB)
            if frame_count % SKIP_FRAMES == 0:
                current_class, current_conf = predict_frame(model, frame_rgb_for_model)

            frame_count += 1

            # C. DIBUJAR INFORMACIÓN
            # Barra negra arriba (ajusta tamaño si cambias resolución)
            cv2.rectangle(frame_display, (0, 0), (FRAME_WIDTH, 50), (0, 0, 0), -1)

            color_texto = COLORS.get(current_class, (255, 255, 255))
            text = f"DETECTADO: {current_class.upper()} ({current_conf*100:.1f}%)"
            cv2.putText(frame_display, text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)

            # Mostrar
            cv2.imshow(WINDOW_NAME, frame_display)

            # Tecla q para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Cerrando...")
        if USING_PICAM:
            try:
                camera.stop()
                camera.close()
            except Exception:
                pass
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
