# Crop the image using the detected object's coordinates
# The image is CT scan of body

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import transforms, datasets

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_name)
        image = self.crop_body_from_image(image)
    
        labels = self.dataframe.loc[idx, config.TARGET_COLS]
        #labels = (labels[:1], labels[1:3], labels[3:6], labels[6:9], labels[9:12])
        #labels = torch.tensor(labels, dtype=torch.float32)
        labels = {
            'bowel': torch.argmax(torch.tensor(labels[:1], dtype=torch.float32)), # binary label
            'extravasation': torch.argmax(torch.tensor(labels[1:3], dtype=torch.float32)), # binary label
            'kidney': torch.argmax(torch.tensor(labels[3:6], dtype=torch.float32)), # multi-class label
            'liver': torch.argmax(torch.tensor(labels[6:9], dtype=torch.float32)), # multi-class label
            'spleen': torch.argmax(torch.tensor(labels[9:12], dtype=torch.float32)), # multi-class label
        }

        if self.transform:
            image_tensor = self.transform(pil_image)


        return image_tensor, labels

    def crop_body_from_image(self, image):
        # Convert to grayscale and then to tensor
        grayscale_image = TF.to_grayscale(image)
        tensor_image = TF.to_tensor(grayscale_image)
        
        # Apply thresholding
        threshold = 0.5  # Adjust based on your CT scan images
        binary_image = (tensor_image > threshold).float()
        
        # Find bounding box of the largest connected component
        labeled_image, num_features = torch.connected_components(binary_image)
        largest_component = torch.bincount(labeled_image.view(-1))[1:].argmax() + 1
        positions = (labeled_image == largest_component).nonzero(as_tuple=True)
        y_min, x_min = positions[0].min(), positions[1].min()
        y_max, x_max = positions[0].max(), positions[1].max()
        
        # Crop the image
        cropped_image = TF.crop(image, y_min, x_min, y_max - y_min, x_max - x_min)
        
        return cropped_image

# Define your own transforms
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a DataFrame with image paths and labels
dataframe = ...

# Create a CustomDataset object with your own transforms
dataset = CustomDataset(dataframe, transform=transform)

# Create a DataLoader object
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)