# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:50:04 2024

@author: stora
"""


import torch
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# Definir el modelo ResNet50 preentrenado en ImageNet
model = models.resnet50(pretrained=True)
model.eval()  # Cambiar a modo evaluación

# Transformaciones: redimensionar, convertir a tensor, normalizar
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para cargar y procesar la imagen
def process_image(image_path):
    img = Image.open(image_path)  # Cargar la imagen con PIL
    img_tensor = preprocess(img)  # Aplicar las transformaciones
    img_tensor = img_tensor.unsqueeze(0)  # Añadir dimensión extra para batch
    return img_tensor

# Función para predecir usando el modelo
def predict(image_path, model):
    img_tensor = process_image(image_path)
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction

# Decodificar las predicciones
def decode_predictions(prediction, top=3):
    # Cargar las etiquetas de ImageNet
    with open("./datos/imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Obtener las probabilidades con softmax
    probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
    
    # Obtener las mejores `top` predicciones
    top_probs, top_idxs = torch.topk(probabilities, top)
    
    decoded = [(labels[idx], prob.item()) for idx, prob in zip(top_idxs, top_probs)]
    return decoded

# Función para mostrar la imagen y las predicciones
def plot_prediction(path, decoded):
    # Tamaño de la figura
    plt.figure(figsize=(8, 8))
    # Cargar imagen con cv2, colores RGB
    photo = cv2.imread(path)
    RGB_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

    # Coordenadas iniciales del texto sobreimpreso
    text_x = 6
    text_y = 20

    # Iteraciones sobre los resultados decoded para sobre imprimir anotaciones
    for label, probability in decoded:
        plt.text(text_x, text_y, 'Predicción: {} - {:.2f}%'.format(label, probability * 100),
                 fontsize=12, color='k', 
                 bbox=dict(boxstyle="round", pad=0.2, fc='white'))
        text_y += 30

    # Mostrar imagen
    plt.imshow(RGB_photo)
    plt.axis('off')
    plt.show()

# Ejemplo de uso:

# Imagen 1
path = 'datos/ball_test.jpg'
prediction = predict(path, model)
decoded = decode_predictions(prediction, top=3)
plot_prediction(path, decoded)

# Imagen 2
path = 'datos/camera_test.jpg'
prediction = predict(path, model)
decoded = decode_predictions(prediction, top=3)
plot_prediction(path, decoded)

# Imagen 3
path = 'datos/moto_test.jpg'
prediction = predict(path, model)
decoded = decode_predictions(prediction, top=3)
plot_prediction(path, decoded)