# Combine multihead multiclass to only one multiclass

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

# Train teset validation split
# labels = dataframe[config.TARGET_COLS].values so that we can use stratify
# To use stratify, we need to the labels are 0,1 and concat them as a string
# and then use treated as a binary number and convert it to decimal
# Then we can use train_test_split with stratify by the decimal number

# Concat all the labels as a string
dataframe['onelabel'] = dataframe[config.TARGET_COLS].apply(lambda x: ''.join(x.astype(str)), axis=1)

# Convert the string to decimal
dataframe['onelabel'] = dataframe['onelabel'].apply(lambda x: int(x, 2))

# dataframe['onelabel'] == 4882 is only one data and it cant split. so conert it 4706
dataframe['onelabel'] = dataframe['onelabel'].apply(lambda x: 4706 if x == 4882 else x)

# conert the 'onelabel' to 0 to N - 1 (N is the number of unique labels)
dataframe['onelabel'] = dataframe['onelabel'].astype('category').cat.codes

# dictionary for the label and the original number
unique_labels = dataframe['onelabel'].unique()
unique_labels.sort()
label_dict = dict(enumerate(unique_labels))
# dictionary for the label and the converted
label_dict = dict(enumerate(dataframe['onelabel'].astype('category').cat.categories))

label_dict = dict(enumerate(dataframe['onelabel'].astype('category').cat.categories))


# Split the data
train_df, test_df = train_test_split(dataframe, test_size=0.5, random_state=42, stratify=dataframe['onelabel'])
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=43, stratify=test_df['onelabel'])
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_name)

        labels = self.dataframe.loc[idx, 'onelabel']
        labels = torch.tensor(labels, dtype=torch.int32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# Denoise the image
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

# Define Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    BilateralFilterTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset and dataloader
train_dataset = CustomDataset(dataframe=train_df, transform=transform)
val_dataset = CustomDataset(dataframe=valid_df, transform=transform)
test_dataset = CustomDataset(dataframe=test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


# Define the model
# I use the pretrained mobilnet_v2 and change the last layer to output 33 classes
# The model is trained on the ImageNet dataset

# Load the pretrained model
model = models.mobilenet_v2(weights='DEFAULT')

# Change the last layer to output 33 classes
num_classes = 33
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Device set to use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# wandb setting
import wandb
mykey = '775553d8bb6bd0180350fae46e7d47bd9ef3f0d9'
if wandb is not None:
    wandb.login(key=mykey)
    run = wandb.init(project="RSNA2023", config={
            "SEED" : config.SEED,
            "BATCH_SIZE" : config.BATCH_SIZE,
            "EPOCHS" : config.EPOCHS,
            "ARCH.": config.MODEL,
            "DENOSE": config.DENOSE,
        })

# Acccuracy, AUROC, Confusion Matrix history
train_acc_history = []
train_auroc_history = []
train_confusion_matrix_history = []
val_acc_history = []
val_auroc_history = []
val_confusion_matrix_history = []


# Train the model
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0.0

    # Metrics for Accuracy and AUROC
    acc = MulticlassAccuracy(num_classes=33)
    auroc = MulticlassAUROC(num_classes=33) 
    confusion_matrix = MulticlassConfusionMatrix(num_classes=33)  

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Metrics
        acc.update(outputs, labels)
        auroc.update(outputs, labels)
        confusion_matrix.update(outputs, labels)

    # Metrics history
    train_acc_history.append(acc.compute())
    train_auroc_history.append(auroc.compute())
    train_confusion_matrix_history.append(confusion_matrix.compute())

    print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {total_loss:.4f}")
    print(f"Train/Accuracy {train_acc_history[-1]}")
    print(f"Train/AUROC {train_auroc_history[-1]}")
    print(f"Train/Confusion Matrix {train_confusion_matrix_history[-1]}")

    # wandb log
    if wandb is not None:
        run.log({
            "epoch": epoch,
            "train_loss": total_loss,
            "train_acc": train_acc_history[-1],
            "train_auroc": train_auroc_history[-1],
            "train_confusion_matrix": train_confusion_matrix_history[-1],
        })

    # Validation
    model.eval()
    total_loss = 0.0

    # Metrics for Accuracy and AUROC
    val_acc = MulticlassAccuracy(num_classes=33)
    val_auroc = MulticlassAUROC(num_classes=33)
    val_confusion_matrix = MulticlassConfusionMatrix(num_classes=33)

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            labels = labels.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Metrics
            val_acc.update(outputs, labels)
            val_auroc.update(outputs, labels)
            val_confusion_matrix.update(outputs, labels)

    # Metrics history
    val_acc_history.append(val_acc.compute())
    val_auroc_history.append(val_auroc.compute())
    val_confusion_matrix_history.append(val_confusion_matrix.compute())


    print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {total_loss:.4f}")
    print(f"Val/Accuracy {val_acc_history[-1]}")
    print(f"Val/AUROC {val_auroc_history[-1]}")
    print(f"Val/Confusion Matrix {val_confusion_matrix_history[-1]}")

    # wandb log
    if wandb is not None:
        run.log({
            "epoch": epoch,
            "val_loss": total_loss,
            "val_acc": val_acc_history[-1],
            "val_auroc": val_auroc_history[-1],
            "val_confusion_matrix": val_confusion_matrix_history[-1],
        })

    print()

# Test
model.eval()
total_loss = 0.0

# Metrics for Accuracy and AUROC
test_acc = MulticlassAccuracy(num_classes=33)
test_auroc = MulticlassAUROC(num_classes=33)
test_confusion_matrix = MulticlassConfusionMatrix(num_classes=33)

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        labels = labels.long().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        # Metrics
        test_acc.update(outputs, labels)
        test_auroc.update(outputs, labels)
        test_confusion_matrix.update(outputs, labels)

test_acc = test_acc.compute()
test_auroc = test_auroc.compute()
test_confusion_matrix = test_confusion_matrix.compute()
print(f"Test/Accuracy {test_acc}")
print(f"Test/AUROC {test_auroc}")
print(f"Test/Confusion Matrix {test_confusion_matrix}")
print()

# wandb log
if wandb is not None:
    run.log({
        "test_loss": total_loss,
        "test_acc": test_acc,
        "test_auroc": test_auroc,
        "test_confusion_matrix": test_confusion_matrix,
    })

# Save the model
torch.save(model.state_dict(), "multihead_model.pth")

# Save the model as onnx
model.eval()
dummy_input = torch.randn(1, 3, 256, 256, device='cuda')
torch.onnx.export(model, dummy_input, "multihead_model.onnx")


import onnx
import onnxruntime
import numpy as np
import torch

# Load the model from onnx
onnx_model = onnx.load("multihead_model.onnx")

# Check the model
onnx.checker.check_model(onnx_model)


# Prediction
# Load the model
model = models.mobilenet_v2(weights='DEFAULT')
num_classes = 33
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("multihead_model.pth"))

# Load the test data
test_df = pd.read_csv(f"{config.BASE_PATH}/test.csv")
test_df["image_path"] = f"{config.BASE_PATH}/test_images"\
                    + "/" + test_df.patient_id.astype(str)\
                    + "/" + test_df.series_id.astype(str)\
                    + "/" + test_df.instance_number.astype(str) +".png"

test_df = test_df.drop_duplicates()

# Create the dataset and dataloader
test_dataset = CustomDataset(dataframe=test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Prediction

# Convert the label to the original label
def convert_label(label):
    # Convert the label to the original label
    label = int(label)
    # Convert the label to the original label
    label = label_dict[label]
    return label

predictions = []
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1)
    pred_indices = probs.argmax(dim=1)
    pred_labels = [convert_label(index.item()) for index in pred_indices]
    pred_binaries = ['{0:b}'.format(label).zfill(13) for label in pred_labels]
    for binary in pred_binaries:
        # split the binary into 13 digits
        binary = list(binary)
        # convert the string to int
        binary = [int(i) for i in binary]
        # append the binary to the predictions
        predictions.append(binary)

predictions_df = pd.DataFrame(predictions, columns=config.TARGET_COLS)

# Check the predictions
# Caluculate the auc of each label

for col in config.TARGET_COLS:
    print(f"{col}: {roc_auc_score(test_df[col], predictions_df[col])}")
