import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from stable_baselines3 import PPO
import warnings

class CNNVisualizer:
    def __init__(self, model_path, force_cpu=True):
        # Load the model
        if force_cpu:
            self.device = torch.device("cpu")
            model = PPO.load(model_path, device=self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PPO.load(model_path)
            
        self.cnn = model.policy.features_extractor.image_net.to(self.device)
        self.cnn.eval()
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])
        
    def get_feature_maps(self, image_path):
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = self.transform(image).unsqueeze(0)
            image = image.to(self.device)
            
            # Store activations
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output.detach())
                
            hooks = []
            for module in self.cnn.modules():
                if isinstance(module, nn.Conv2d):
                    hooks.append(module.register_forward_hook(hook_fn))
                    
            with torch.no_grad():
                _ = self.cnn(image)
                
            for hook in hooks:
                hook.remove()
                
            return activations
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def visualize_feature_maps(self, image_path, save_dir):
        activations = self.get_feature_maps(image_path)
        if activations is None:
            return
            
        for i, layer_activations in enumerate(activations):
            n_features = layer_activations.shape[1]
            size = int(np.ceil(np.sqrt(n_features)))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, axes = plt.subplots(size, size, figsize=(12, 12))
                
                if size == 1:
                    axes = np.array([[axes]])
                elif len(axes.shape) == 1:
                    axes = axes.reshape(size, size)
                
                for idx in range(n_features):
                    ax = axes[idx//size, idx%size]
                    ax.imshow(layer_activations[0, idx].cpu().numpy(), cmap='viridis')
                    ax.axis('off')
                
                for idx in range(n_features, size*size):
                    axes[idx//size, idx%size].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'layer_{i+1}_features.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

    def visualize_filters(self, save_dir):
        for i, module in enumerate(self.cnn.modules()):
            if isinstance(module, nn.Conv2d):
                filters = module.weight.data.cpu()
                n_filters = filters.shape[0]
                size = int(np.ceil(np.sqrt(n_filters)))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig, axes = plt.subplots(size, size, figsize=(12, 12))
                    
                    if size == 1:
                        axes = np.array([[axes]])
                    elif len(axes.shape) == 1:
                        axes = axes.reshape(size, size)
                    
                    for idx in range(n_filters):
                        ax = axes[idx//size, idx%size]
                        filter_img = filters[idx].mean(0)
                        ax.imshow(filter_img.numpy(), cmap='viridis')
                        ax.axis('off')
                    
                    for idx in range(n_filters, size*size):
                        axes[idx//size, idx%size].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'conv_layer_{i}_filters.png')
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

def main():
    model_path = r"E:\PPO_BC_MODELS\models_ppo_large\best_model.zip"
    save_dir = "cnn_visualizations"
    image_dir = "test_images"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading model and initializing visualizer...")
    visualizer = CNNVisualizer(model_path, force_cpu=True)
    
    print("Visualizing CNN filters...")
    visualizer.visualize_filters(save_dir)
    
    if os.path.exists(image_dir):
        print(f"Processing images from {image_dir}...")
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing image {i}/{len(image_files)}: {image_file}")
            image_path = os.path.join(image_dir, image_file)
            img_save_dir = os.path.join(save_dir, os.path.splitext(image_file)[0])
            os.makedirs(img_save_dir, exist_ok=True)
            
            visualizer.visualize_feature_maps(image_path, img_save_dir)
    else:
        print(f"Image directory {image_dir} not found!")

    print(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    main()
