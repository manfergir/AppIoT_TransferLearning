# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:23:10 2024

@author: stora
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.datasets as datasets
# pip install torchinfo
from torchinfo import summary

import numpy as np
import h5py
import matplotlib.pyplot as plt



# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Preprocesamiento estilo ResNet (imagenet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Cargamos los datos
train_dir = "./pizza_steak_sushi/train"
test_dir = "./pizza_steak_sushi/test"

# Dataset para imágenes en PyTorch
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get class names as a list
class_names = train_dataset.classes
class_names

# Visualización de una imagen
imagen, etiqueta = train_dataset[0]

# Desnormalizar aplicando (imagen * std) + mean para cada canal (R, G, B)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # Redimensionar para broadcast
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Aplicar la desnormalización: (imagen * std) + mean
imagen = imagen * std + mean

# Cambiar la forma del tensor de (C, H, W) a (H, W, C) para poder mostrarlo
imagen = imagen.permute(1, 2, 0)

numpy_image = imagen.numpy()
# Mostrar la imagen
plt.imshow(numpy_image)
plt.axis('off')  # Quitar los ejes si quieres
plt.show()

#%%

# Definir modelo con ResNet50 preentrenado:
# Cargar ResNet50 preentrenado en PyTorch
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')


# # Imprime un resumen utilizando torchinfo 
summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# Congelar todas las capas excepto las de batch normalization
for name, param in model.named_parameters():
    if 'bn' not in name:  # 'bn' se refiere a las capas BatchNorm
        param.requires_grad = False  # Congelar todas las demás capas
    else:
        param.requires_grad = True   # Asegurar que BatchNorm sea entrenable

# Modificar la última capa de ResNet50 para nuestro número de clases (6)
num_ftrs = model.fc.in_features




# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2), 
    torch.nn.Linear(in_features=num_ftrs, 
                    out_features=output_shape, # same number of output units as our number of classes
                    ))


# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Entrenamiento y evaluación:
# Definir optimizador y función de pérdida
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Entrenamiento
num_epochs = 60
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        running_corrects += torch.sum(outputs.argmax(dim=1) == labels)
    
    train_loss = running_loss / len(train_dataset)
    train_accuracy = 100.0 * running_corrects.item() / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    
    # Validación
    model.eval() 
    corrects = 0
    val_loss = 0.0
    total = 0

    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += loss.item()
            total += labels.size(0)
            corrects += torch.sum(outputs.argmax(dim=1) == labels)
    
    val_loss /= len(test_loader)
    val_accuracy = 100.0 * corrects.item() / total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_accuracy)  
    test_acc = corrects.double() / len(test_dataset)
    
    
 
    print(f'Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - Test Loss: {val_loss:.4f} - Test Accuracy: {test_acc:.4f}' )

# Gráficas de evolución de Accuracy y Loss
epochs = range(1, num_epochs+1)

# Training and validation accuracy
plt.plot(epochs, train_acc_history, label='Train Accuracy')
plt.plot(epochs, val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

# Training and validation loss
plt.plot(epochs, train_loss_history, label='Train Loss')
plt.plot(epochs, val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

    
