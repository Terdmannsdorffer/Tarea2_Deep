import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys

class ImageEncoders:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize preprocessing for ResNet and DINOv2
        self.preprocess_resnet_dino = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ])
        
        # Models will be loaded on demand
        self.resnet_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.dinov2_model = None
    
    def load_resnet(self, version=34):
        """Load ResNet model with specified version (18, 34, 50, etc.)"""
        print(f"Loading ResNet{version}...")
        if version == 18:
            self.resnet_model = models.resnet18(weights='IMAGENET1K_V1').to(self.device)
        elif version == 34:
            self.resnet_model = models.resnet34(weights='IMAGENET1K_V1').to(self.device)
        elif version == 50:
            self.resnet_model = models.resnet50(weights='IMAGENET1K_V1').to(self.device)
        else:
            raise ValueError(f"ResNet{version} not supported. Try 18, 34, or 50.")
        
        # Cancel the classification layer
        self.resnet_model.fc = torch.nn.Identity()
        self.resnet_model.eval()
        return self.resnet_model
    
    def load_clip(self, model_name="ViT-B/32"):
        """Load CLIP model with specified name"""
        print(f"Loading CLIP {model_name}...")
        
        import clip
        model, preprocess = clip.load(model_name, device=self.device)
        self.clip_model = model.encode_image
        self.clip_preprocess = preprocess
        return self.clip_model, self.clip_preprocess
    
    def load_dinov2(self, version='vits14'):
        """Load DINOv2 model with specified version"""
        print(f"Loading DINOv2 {version}...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{version}').to(self.device)
        self.dinov2_model.eval()
        return self.dinov2_model
    
    def extract_features(self, image_path, model_type, specific_model=None):
        """
        Extract features from an image using the specified model
        
        Parameters:
        - image_path: Path to the image file
        - model_type: One of 'resnet', 'clip', or 'dinov2'
        - specific_model: Specific model version (e.g., 34 for ResNet34)
        
        Returns:
        - features: Extracted features tensor
        - dimension: Feature dimension
        """
        # Open and prepare image
        image = Image.open(image_path).convert('RGB')
        
        if model_type.lower() == 'resnet':
            # Load model if not already loaded
            if self.resnet_model is None:
                version = 34 if specific_model is None else specific_model
                self.load_resnet(version)
            
            # Preprocess image
            processed_image = self.preprocess_resnet_dino(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.resnet_model(processed_image)
        
        elif model_type.lower() == 'clip':
            # Load model if not already loaded
            if self.clip_model is None or self.clip_preprocess is None:
                model_name = "ViT-B/32" if specific_model is None else specific_model
                self.load_clip(model_name)
            
            # Preprocess image
            processed_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.clip_model(processed_image)
        
        elif model_type.lower() == 'dinov2':
            # Load model if not already loaded
            if self.dinov2_model is None:
                version = 'vits14' if specific_model is None else specific_model
                self.load_dinov2(version)
            
            # Preprocess image
            processed_image = self.preprocess_resnet_dino(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.dinov2_model(processed_image)
        
        else:
            raise ValueError("Model type not recognized. Choose 'resnet', 'clip', or 'dinov2'")
        
        # Get feature dimension
        dimension = features.shape[1]
        
        return features, dimension