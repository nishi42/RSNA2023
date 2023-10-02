# Crop the image using the detected object's coordinates
# The image is CT scan of body

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor

class BilateralFilterTransform:
    def __init__(self, d=15, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def __call__(self, image):
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        denoised_image = cv2.bilateralFilter(image_np, self.d, self.sigmaColor, self.sigmaSpace)
        # Convert back to PIL Image
        return ToPILImage()(denoised_image)

class OriginalCrop:
    def __init__(self):
        pass
    def __call__(self, image, bbox):
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        # Crop the image
        x1, y1, x2, y2 = bbox
        cropped_image = image_np[y1:y2, x1:x2]
        # Convert back to PIL Image
        return ToPILImage()(cropped_image)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])