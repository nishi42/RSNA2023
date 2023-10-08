import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import cv2
import onnx
import onnxruntime

# Set up the configuration
class Config:
    SEED = 42
    IMAGE_SIZE = [256, 256]
    BATCH_SIZE = 32
    EPOCHS = 10
    TARGET_COLS  = [
        "bowel_healthy","bowel_injury", "extravasation_healthy","extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]
    BASE_PATH = "/kaggle/input/rsna-atd-512x512-png-v2-dataset"
    MODEL = "mobilenet_v2"
    CROP = "CenterCROP"
    DENOSE = 'BilateralFilterTransform'

config = Config()

# Load the data
dataframe = pd.read_csv(f"{config.BASE_PATH}/train.csv")
dataframe["image_path"] = f"{config.BASE_PATH}/train_images"\
                    + "/" + dataframe.patient_id.astype(str)\
                    + "/" + dataframe.series_id.astype(str)\
                    + "/" + dataframe.instance_number.astype(str) +".png"
dataframe = dataframe.drop_duplicates()

class PatientDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        # Add any other necessary initializations or transformations here

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load your image and perform necessary transformations
        image = Image.open(self.image_paths[idx])  # Define this function

        if self.transform:
            image = self.transform(image)

        return to_numpy(image)

# Define the transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
onnx_model = onnx.load(config.BASE_PATH + "/model.onnx")
ort_session = onnxruntime.InferenceSession(config.BASE_PATH + "/model.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Create the datasets
infer_dataset = CustomDataset(dataframe=train_df, transform=transform)

# Define the dataloader
data_loader = DataLoader(infer_dataset, batch_size=256, shuffle=True)

# Getting unique patient IDs from test dataset
patient_ids = test_df["patient_id"].unique()

# Initializing array to store predictions
patient_preds = np.zeros(
    shape=(len(patient_ids), 2*2 + 3*3),
    dtype="float32"
)

# Iterating over each patient
for pidx, patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids), desc="Patients "):
    print(f"Patient ID: {patient_id}")
    
    # Query the dataframe for a particular patient
    patient_df = test_df.query("patient_id == @patient_id")
    
    # Getting image paths for a patient
    patient_paths = patient_df.image_path.tolist()

    # Create the datasets
    infer_dataset = CustomDataset(dataframe=patient_df, transform=transform)

    # Define the dataloader
    data_loader = DataLoader(infer_dataset, batch_size=1, shuffle=Fals)
    
    # Building dataset for prediction
    dtest = PatientDataset(patient_paths, sub_transform)
    dataloader = DataLoader(dtest, shuffle=False)  # Adjust batch_size if necessary
    
    # Predicting with the model
    preds = []
    for images in dataloader:
        # Assuming your model is on the same device as your data
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        ort_outs = ort_session.run(None, ort_inputs)
        # Apply softmax to each tensor in the outputs
        softmaxed_outputs = [softmax(_) for _ in ort_outs]
        preds.append([np.argmax(_) for _ in softmaxed_outputs])


    # Get the output and split it into 5 parts
    pred = [F.softmax(torch.tensor(output)) for output in ort_outs[:5]]
    pred = np.concatenate(pred, axis=-1).astype("float32")
    pred = pred[:len(patient_paths), :]
    pred = np.mean(pred.reshape(1, len(patient_paths), 11), axis=0)
    pred = np.max(pred, axis=0, keepdims=True)
    
    patient_preds[pidx, :] += post_proc(pred)[0]
    
    # Deleting variables to free up memory 
    del patient_df, patient_paths, dtest, pred; gc.collect()
