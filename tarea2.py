import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import sys

# Importar nuestro módulo de encoders
from encoders import ImageEncoders

# Definir clases del dataset VOC-Pascal
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class VOCDataset(Dataset):
    def __init__(self, root_dir, list_file, encoder_type, transform=None):
        """
        Dataset para VOC-Pascal
        
        Args:
            root_dir: Directorio raíz con imágenes
            list_file: Archivo que contiene la lista de imágenes y etiquetas
            encoder_type: Tipo de encoder a utilizar ('resnet', 'dinov2', 'clip')
            transform: Transformaciones a aplicar a las imágenes
        """
        self.root_dir = root_dir
        self.encoder_type = encoder_type
        self.transform = transform
        
        # Leer archivo con la lista de imágenes y etiquetas
        self.data = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:  # Asegurarse de que tenga todos los campos necesarios
                    image_id, class_name, xmax, xmin, ymax, ymin = parts[:6]
                    self.data.append({
                        'image_id': image_id,
                        'class': class_name,
                        'xmax': int(xmax),
                        'xmin': int(xmin),
                        'ymax': int(ymax),
                        'ymin': int(ymin),
                        'class_id': VOC_CLASSES.index(class_name)
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.root_dir, 'JPEGImages', f"{item['image_id']}.jpg")
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Recortar la imagen al rectángulo del objeto
            image = image.crop((item['xmin'], item['ymin'], item['xmax'], item['ymax']))
            
            if self.transform:
                image = self.transform(image)
                
            return {
                'image': image,
                'class_id': item['class_id'],
                'class_name': item['class'],
                'image_id': item['image_id']
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Devolver un placeholder en caso de error
            if self.transform:
                return {
                    'image': torch.zeros(3, 224, 224),
                    'class_id': item['class_id'],
                    'class_name': item['class'],
                    'image_id': item['image_id']
                }


class KNNClassifier:
    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        n_test = X_test.shape[0]
        y_pred = np.zeros(n_test, dtype=self.y_train.dtype)
        
        for i in range(n_test):
            # Calcular distancias euclideas
            distances = np.sqrt(np.sum((self.X_train - X_test[i])**2, axis=1))
            
            # Encontrar los k vecinos más cercanos
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Voto mayoritario
            y_pred[i] = np.bincount(k_nearest_labels).argmax()
        
        return y_pred


class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def extract_features(dataset, encoder, device, batch_size=32):
    """Extrae características usando el encoder especificado"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extrayendo características"):
            images = batch['image'].to(device)
            
            # Extraemos características según el tipo de encoder
            batch_features = encoder(images)
                
            features.append(batch_features.cpu().numpy())
            labels.append(batch['class_id'].numpy())
    
    return np.vstack(features), np.concatenate(labels)


def train_model(model, X_train, y_train, device, num_epochs=30, batch_size=64):
    """Entrena un modelo neuronal"""
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Crear mini-batches
        indices = torch.randperm(X_train_tensor.size(0))
        
        total_loss = 0
        num_batches = 0
        
        for i in range(0, X_train_tensor.size(0), batch_size):
            # Obtener mini-batch
            idx = indices[i:min(i + batch_size, X_train_tensor.size(0))]
            X_batch = X_train_tensor[idx]
            y_batch = y_train_tensor[idx]
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / num_batches:.4f}')
    
    return model


def evaluate_model(model, X_test, y_test, device):
    """Evalúa un modelo neuronal"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
    
    accuracy = accuracy_score(y_test, predicted)
    conf_matrix = confusion_matrix(y_test, predicted)
    
    # Calcular accuracy por clase
    class_accuracy = {}
    for class_idx in range(len(VOC_CLASSES)):
        class_mask = (y_test == class_idx)
        if np.sum(class_mask) > 0:  # Si hay ejemplos de esta clase
            class_accuracy[VOC_CLASSES[class_idx]] = accuracy_score(
                y_test[class_mask], predicted[class_mask]
            )
    
    return accuracy, class_accuracy, conf_matrix, predicted


def plot_confusion_matrix(conf_matrix, classes, title):
    """Grafica la matriz de confusión"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


def plot_class_accuracy(class_accuracies, title):
    """Grafica el accuracy por clase para diferentes modelos"""
    df = pd.DataFrame(class_accuracies)
    df = df.reindex(VOC_CLASSES)  # Ordenar por clases
    
    plt.figure(figsize=(15, 8))
    df.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Clase')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modelo')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluación de encoders visuales para clasificación de imágenes')
    parser.add_argument('--data_dir', type=str, default='VocPascal', help='Directorio de datos')
    parser.add_argument('--encoder', type=str, default='all', choices=['all', 'resnet', 'dinov2', 'clip'],
                        help='Tipo de encoder a utilizar')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Inicializar encoders
    encoders_manager = ImageEncoders(device)
    
    # Definir transformaciones para cada encoder
    transform_resnet_dino = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preparar encoders a evaluar
    encoders_to_evaluate = []
    if args.encoder == 'all' or args.encoder == 'resnet':
        resnet_model = encoders_manager.load_resnet(34)
        encoders_to_evaluate.append(('resnet', resnet_model, transform_resnet_dino, 512))
    
    if args.encoder == 'all' or args.encoder == 'dinov2':
        dinov2_model = encoders_manager.load_dinov2('vits14')
        encoders_to_evaluate.append(('dinov2', dinov2_model, transform_resnet_dino, 384))
    
    if args.encoder == 'all' or args.encoder == 'clip':
        clip_model, clip_preprocess = encoders_manager.load_clip("ViT-B/32")
        encoders_to_evaluate.append(('clip', clip_model, clip_preprocess, 512))
    
    # Resultados para graficar
    all_class_accuracies = {}
    best_model_info = {'accuracy': 0, 'encoder': None, 'model': None, 'conf_matrix': None}
    
    # Evaluar cada encoder
    for encoder_name, encoder, transform, feature_dim in encoders_to_evaluate:
        print(f"\n{'='*50}\nEvaluando encoder: {encoder_name.upper()}\n{'='*50}")
        
        # Cargar datasets
        train_dataset = VOCDataset(
            root_dir=args.data_dir,
            list_file=os.path.join(args.data_dir, 'train_voc.txt'),
            encoder_type=encoder_name,
            transform=transform
        )
        
        val_dataset = VOCDataset(
            root_dir=args.data_dir,
            list_file=os.path.join(args.data_dir, 'val_voc.txt'),
            encoder_type=encoder_name,
            transform=transform
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Extraer características
        print("\nExtrayendo características de entrenamiento...")
        X_train, y_train = extract_features(train_dataset, encoder, device)
        
        print("\nExtrayendo características de validación...")
        X_test, y_test = extract_features(val_dataset, encoder, device)
        
        # Normalizar características para los métodos kNN
        X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
        
        # Evaluar modelos
        models_to_evaluate = [
            ('1-NN', KNNClassifier(k=1)),
            ('5-NN', KNNClassifier(k=5)),
            ('Linear', LinearClassifier(feature_dim, len(VOC_CLASSES)).to(device)),
            ('MLP', MLPClassifier(feature_dim, 256, len(VOC_CLASSES)).to(device))
        ]
        
        for model_name, model in models_to_evaluate:
            print(f"\n{'-'*30}\nEvaluando {encoder_name} con {model_name}\n{'-'*30}")
            
            # Entrenar el modelo
            if model_name in ['1-NN', '5-NN']:
                model.fit(X_train_norm, y_train)
                y_pred = model.predict(X_test_norm)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calcular accuracy por clase
                class_accuracy = {}
                for class_idx in range(len(VOC_CLASSES)):
                    class_mask = (y_test == class_idx)
                    if np.sum(class_mask) > 0:
                        class_accuracy[VOC_CLASSES[class_idx]] = accuracy_score(
                            y_test[class_mask], y_pred[class_mask]
                        )
                
                conf_matrix = confusion_matrix(y_test, y_pred)
                
            else:  # Linear o MLP
                print("Entrenando modelo...")
                model = train_model(model, X_train, y_train, device)
                accuracy, class_accuracy, conf_matrix, y_pred = evaluate_model(model, X_test, y_test, device)
            
            # Guardar resultados
            print(f"Accuracy total: {accuracy:.4f}")
            
            # Guardar accuracy por clase para graficar
            model_key = f"{encoder_name}_{model_name}"
            all_class_accuracies[model_key] = class_accuracy
            
            # Actualizar mejor modelo
            if accuracy > best_model_info['accuracy']:
                best_model_info = {
                    'accuracy': accuracy,
                    'encoder': encoder_name,
                    'model': model_name,
                    'conf_matrix': conf_matrix,
                    'class_accuracies': class_accuracy
                }
    
    # Graficar accuracy por clase para diferentes modelos
    # Convertir diccionario a formato adecuado para graficar
    plot_data = {}
    for model_key, class_acc in all_class_accuracies.items():
        for class_name, acc in class_acc.items():
            if class_name not in plot_data:
                plot_data[class_name] = {}
            plot_data[class_name][model_key] = acc
    
    # Graficar matriz de confusión del mejor modelo
    if best_model_info['conf_matrix'] is not None:
        plot_confusion_matrix(
            best_model_info['conf_matrix'], 
            VOC_CLASSES,
            f"Matriz de Confusión - {best_model_info['encoder']} + {best_model_info['model']}"
        )
        
        # Graficar accuracy por clase para cada encoder
        for encoder_name, _, _, _ in encoders_to_evaluate:
            encoder_results = {k: v for k, v in all_class_accuracies.items() if k.startswith(f"{encoder_name}_")}
            
            # Convertir a DataFrame para facilitar el gráfico
            encoder_df = {}
            for model_key, class_acc in encoder_results.items():
                model_name = model_key.split('_')[1]
                encoder_df[model_name] = pd.Series(class_acc)
            
            df = pd.DataFrame(encoder_df)
            
            plt.figure(figsize=(12, 8))
            df.plot(kind='bar')
            plt.title(f"Accuracy por Clase - {encoder_name.upper()}")
            plt.ylabel('Accuracy')
            plt.xlabel('Clase')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Modelo')
            plt.tight_layout()
            plt.savefig(f"accuracy_by_class_{encoder_name}.png")
            plt.close()
    
    print("\n\nResumen de resultados:")
    print(f"Mejor modelo: {best_model_info['encoder']} + {best_model_info['model']}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    
    # Generar informe de resultados
    with open('resultados.txt', 'w') as f:
        f.write("Resultados de la evaluación de encoders visuales\n")
        f.write("="*50 + "\n\n")
        
        f.write("Mejor modelo: " + best_model_info['encoder'] + " + " + best_model_info['model'] + "\n")
        f.write(f"Accuracy: {best_model_info['accuracy']:.4f}\n\n")
        
        f.write("Accuracy por clase del mejor modelo:\n")
        for class_name, acc in best_model_info['class_accuracies'].items():
            f.write(f"{class_name}: {acc:.4f}\n")


if __name__ == "__main__":
    main()