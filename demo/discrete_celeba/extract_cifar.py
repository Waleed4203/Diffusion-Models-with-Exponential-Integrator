import os
import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

def main():
    # 1. Create a directory to store real CIFAR-10 images
    real_images_dir = 'temp/cifar10_real'
    os.makedirs(real_images_dir, exist_ok=True)
    
    print("Downloading/Loading CIFAR-10 dataset...")
    # This will automatically use your downloaded archive if it's in the data folder, 
    # or it will download it quickly.
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    
    print(f"Extracting {len(dataset)} images to {real_images_dir}...")
    for i, (img, label) in enumerate(tqdm(dataset)):
        save_image(img, os.path.join(real_images_dir, f'{i}.png'))
        
    print("\nExtraction complete! Now you can calculate the stats or FID.")

if __name__ == '__main__':
    main()
