# diagnóstico_picam2_color.py
import cv2
import numpy as np
from PIL import Image
import time
try:
    from picamera2 import Picamera2
    import libcamera
    USING_PICAM = True
except Exception:
    USING_PICAM = False
    print("No hay Picamera2; pruebe igualmente con su cámara USB.")

W = 640
H = 480

def show_and_save(name, img):
    cv2.imshow(name, img)
    cv2.imwrite(f"{name}.png", img)

def mean_channels(arr):
    # arr expected HWC
    m = arr.mean(axis=(0,1))
    return m

if __name__ == "__main__":
    if USING_PICAM:
        cam = Picamera2()
        # Intenta pedir RGB888 (como hiciste antes). No forceamos BGR para diagnóstico.
        cfg = cam.create_still_configuration(main={"size": (W, H), "format": "RGB888"})
        cam.configure(cfg)
        cam.start()
        time.sleep(0.5)
        frame = cam.capture_array("main")
        cam.stop()
        cam.close()
        if frame is None:
            raise SystemExit("capture_array devolvió None")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise SystemExit("No se pudo leer la cámara USB")

    # frame está tal cual devuelve la API. Puede ser RGB u BGR.
    # A: guardamos y mostramos el raw tal cual (lo que devuelve Picamera2)
    raw = frame.copy().astype(np.uint8)

    # B: suponiendo raw es BGR (lo mostramos tal cual)
    bgr_assumed = raw.copy()
    # C: suponiendo raw es RGB (convertimos a BGR para mostrar con OpenCV)
    rgb_assumed = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    # Guardar versiones para inspección externa
    # Usamos PIL para guardar la versión que creemos RGB (para abrir con otros visores)
    try:
        Image.fromarray(cv2.cvtColor(bgr_assumed, cv2.COLOR_BGR2RGB)).save("saved_from_bgr_assumed_as_rgb.png")
    except Exception:
        pass
    try:
        Image.fromarray(cv2.cvtColor(rgb_assumed, cv2.COLOR_BGR2RGB)).save("saved_from_rgb_assumed_as_rgb.png")
    except Exception:
        pass
    # También guardamos el raw interpretado como bytes directos (puede verse mal si el orden es distinto)
    try:
        Image.fromarray(raw).save("saved_raw_direct.png")
    except Exception:
        pass

    # Imprimimos medias por canal (orden tal cual en el array)
    m = mean_channels(raw)
    print("Medias por canal del array raw (canal0, canal1, canal2):", m)

    # Imprimir medias interpretadas
    print("Medias suponiendo BGR -> (B,G,R):", mean_channels(bgr_assumed))
    print("Medias suponiendo RGB -> mostrado BGR:", mean_channels(rgb_assumed))

    # Mostrar ventanas con las 3 opciones para comparar rápidamente
    cv2.namedWindow("raw_as_given", cv2.WINDOW_NORMAL)
    cv2.namedWindow("assume_BGR_show", cv2.WINDOW_NORMAL)
    cv2.namedWindow("assume_RGB_show", cv2.WINDOW_NORMAL)

    cv2.imshow("raw_as_given", raw)
    cv2.imshow("assume_BGR_show", bgr_assumed)
    cv2.imshow("assume_RGB_show", rgb_assumed)

    print("Se han creado archivos: saved_raw_direct.png, saved_from_bgr_assumed_as_rgb.png, saved_from_rgb_assumed_as_rgb.png")
    print("Observa las 3 ventanas. Pulsa cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
