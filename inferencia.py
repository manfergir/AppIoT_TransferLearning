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
# Helpers captura
# =========================
def picam_frame_to_display_and_rgb(frame_cam):
    """
    Recibe frame_cam tal cual devuelve Picamera2 (uint8 HWC).
    Devuelve (frame_display_bgr, frame_rgb_for_model).
    Heurística: Picamera2 a veces devuelve BGR aun pidiendo RGB888.
    Detectamos mediante medias por canal y adaptamos sin invertir dos veces.
    """
    # Copia para no modificar original
    frm = frame_cam.copy()

    # Calculamos medias por canal (canal 0,1,2 según arreglo)
    means = frm.mean(axis=(0,1))  # array([m0, m1, m2])
    # Interpretación:
    # - Si asumimos que el orden del array es [C0, C1, C2]:
    #   * si C0 >> C2 (ej. C0 > 1.1 * C2) es muy probable que C0 sea Blue (BGR)
    #   * si C2 >> C0 es probable que C2 sea Blue (RGB order -> R in index 0 ?)
    # En la práctica comprobamos si la imagen "parece azul" por exceso en canal 0.
    b, g, r = means[0], means[1], means[2]

    # Caso más frecuente: si C0 (canal 0) es claramente mayor que C2 => ya es BGR
    if b > r * 1.1:
        # frm está en BGR: lo dejamos así para mostrar y convertimos a RGB para el modelo
        frame_display = frm.copy()
        frame_rgb_for_model = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
    elif r > b * 1.1:
        # frm tiene más rojo en canal 0 (r >> b) -> probablemente está en RGB ordenado
        # Convertimos a BGR para mostrar y generamos RGB para el modelo
        frame_rgb_for_model = frm.copy()
        frame_display = cv2.cvtColor(frame_rgb_for_model, cv2.COLOR_RGB2BGR)
    else:
        # Caso ambiguo: asumimos que está en BGR (comportamiento más habitual)
        frame_display = frm.copy()
        frame_rgb_for_model = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)

    return frame_display, frame_rgb_for_model

# =========================
# 5. MAIN
# =========================
def main():
    model = load_model()

    if USING_PICAM:
        camera = Picamera2()
        config = camera.create_still_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        print("📷 Cámara Pi iniciada (Configuración Still).")
    else:
        cap = cv2.VideoCapture(0)
        # Intentamos forzar resolución USB al tamaño objetivo
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_count = 0
    current_class = "Esperando..."
    current_conf = 0.0

    COLORS = {
        "casco": (0, 255, 0),        # BGR para dibujar (verde)
        "mascarilla": (255, 255, 0), # BGR (amarillo)
        "nada": (0, 0, 255)          # BGR (rojo)
    }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            # A. CAPTURA
            if USING_PICAM:
                frame_cam = camera.capture_array("main")  # lo que devuelve Picamera2
                # Robustecer: transformar a frame_display (BGR) y frame_rgb_for_model (RGB)
                frame_display, frame_rgb_for_model = picam_frame_to_display_and_rgb(frame_cam)
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
            except Exception:
                pass
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
