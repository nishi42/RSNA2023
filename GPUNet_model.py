# Read the data and train a simple model

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
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

# Set up the configuration
class Config:
    SEED = 42
    IMAGE_SIZE = [512, 512]
    BATCH_SIZE = 256
    EPOCHS = 5
    TARGET_COLS  = [
        "bowel_healthy","bowel_injury", "extravasation_healthy","extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]
    BASE_PATH = "/content/drive/MyDrive/kaggle/RSNA2023"
    HALF = True
    MODEL = "GPUNet"
    DENOISE = False
    


config = Config()

# Load the data
dataframe = pd.read_csv(f"{config.BASE_PATH}/train.csv")
dataframe["image_path"] = f"{config.BASE_PATH}/train_images"\
                    + "/" + dataframe.patient_id.astype(str)\
                    + "/" + dataframe.series_id.astype(str)\
                    + "/" + dataframe.instance_number.astype(str) +".png"
dataframe = dataframe.drop_duplicates()

# Train teset validation split

# Split the data 
sgkf = StratifiedGroupKFold(n_splits=2, random_state=42)
# train test split
groups = dataframe['patient_id']
for i, (train_index, test_index) in enumerate(sgkf.split(X = dataframe, y = None, groups = groups)):
    train_df, test_df = dataframe[train_index], dataframe[test_index]
# test and validation split
groups = X_test['patient_id']
for i, (train_index, test_index) in enumerate(sgkf.split(X = test_df, y = None, groups = groups)):
    val_df, test_df = test_df[train_index], test_df[test_index]

# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_name)

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
            image = self.transform(image)

        return image, labels

# Define the transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Create the datasets
train_dataset = CustomDataset(dataframe=train_df, transform=transform)
val_dataset = CustomDataset(dataframe=valid_df, transform=transform)
test_dataset = CustomDataset(dataframe=test_df, transform=transform)

# Define the dataloader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define the model
class MultiHeadGPUNet(nn.Module):
    def __init__(self):
        super(MultiHeadGPUNet, self).__init__()

        # Load the pretrained GPU model
        model_type = "GPUNet-0" # select one from above
        precision = "fp32" # select either fp32 of fp16 (for better performance on GPU)
        self.base_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type=model_type, model_math=precision)

        # Freeze the parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the existing classifier with your own multi-head output
        out_features = 1000

        # Define the new classifiers (heads)
        self.head1 = nn.Linear(out_features, 1)  # binary label
        self.head2 = nn.Linear(out_features, 1)  # binary label
        self.head3 = nn.Linear(out_features, 3)  # multi-class label
        self.head4 = nn.Linear(out_features, 3)  # multi-class label
        self.head5 = nn.Linear(out_features, 3)  # multi-class label

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten

        out1 = self.head1(x)
        out2 = self.head2(x)
        out3 = self.head3(x)
        out4 = self.head4(x)
        out5 = self.head5(x)

        return out1, out2, out3, out4, out5



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MultiHeadGPUNet()
model = model.to(device)

# Define the loss function and optimizer
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# wandb setting
import wandb
mykey = '775553d8bb6bd0180350fae46e7d47bd9ef3f0d9'
# wandb = None
if wandb is not None:
    wandb.login(key=mykey)
    run = wandb.init(project="RSNA2023", config={
            "SEED" : config.SEED,
            "BATCH_SIZE" : config.BATCH_SIZE,
            "EPOCHS" : config.EPOCHS,
            "ARCH.": config.MODEL,
            "DENOISE": config.DENOISE,
        })


# Acccuracy, AUROC
train_acc_history = []
train_auroc_history = []
val_acc_history = []
val_auc_history = []

# Train the model
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0.0

    # Metrics for Accuracy and AUROC
    acc_bowel = MulticlassAccuracy(num_classes=2)
    acc_extravasation =MulticlassAccuracy(num_classes=2)
    acc_kidney = MulticlassAccuracy(num_classes=3)
    acc_liver = MulticlassAccuracy(num_classes=3)
    acc_spleen = MulticlassAccuracy(num_classes=3)
    auc_bowel = MulticlassAUROC(num_classes=2)
    auc_extravasation = MulticlassAUROC(num_classes=2)
    auc_kidney = MulticlassAUROC(num_classes=3)
    auc_liver = MulticlassAUROC(num_classes=3)
    auc_spleen = MulticlassAUROC(num_classes=3)

    for i, (inputs, labels_dict) in enumerate(tqdm(train_loader)):

        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
        labels_list = [labels_dict[key].to(device) for key in labels_dict]

        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Calculate each output loss
        loss1 = criterion1(outputs[0], labels_list[0][0])
        loss2 = criterion1(outputs[1], labels_list[1][0])
        loss3 = criterion2(outputs[2], labels_list[2])
        loss4 = criterion2(outputs[3], labels_list[3])
        loss5 = criterion2(outputs[4], labels_list[4])
        # Sum all the losses
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate the accuracy
        acc_bowel.update(outputs[0].cpu(), labels_list[0][0].cpu())
        acc_extravasation.update(outputs[1].cpu(), labels_list[1][0].cpu())
        acc_kidney.update(outputs[2].cpu(), labels_list[2].cpu())
        acc_liver.update(outputs[3].cpu(), labels_list[3].cpu())
        acc_spleen.update(outputs[4].cpu(), labels_list[4].cpu())


        # Calculate the AUROC for each head
        auc_bowel.update(outputs[0].cpu(), labels_list[0][0].cpu())
        auc_extravasation.update(outputs[1].cpu(), labels_list[1][0].cpu())
        auc_kidney.update(outputs[2].cpu(), labels_list[2].cpu())
        auc_liver.update(outputs[3].cpu(), labels_list[3].cpu())
        auc_spleen.update(outputs[4].cpu(), labels_list[4].cpu())
        

    # Add to the history
    train_acc_history.append({"EPOCH": epoch,
                              "bowel": acc_bowel.compute(),
                              "extravasation": acc_extravasation.compute(),
                              "kidney": acc_kidney.compute(),
                              "liver": acc_liver.compute(),
                              "spleen": acc_spleen.compute()})
    train_auroc_history.append({"EPOCH": epoch,
                                "bowel": auc_bowel.compute(),
                                "extravasation": auc_extravasation.compute(),
                                "kidney": auc_kidney.compute(),
                                "liver": auc_liver.compute(),
                                "spleen": auc_spleen.compute()})

    print({'EPOCH': epoch})
    print({'head': 'Train/Accuracy_head_0', 'acc': acc_bowel.compute(), 'epoch': epoch})
    print({'head': 'Train/AUC_head_0', 'auc': auc_bowel.compute(), 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_1', 'acc': acc_extravasation.compute(), 'epoch': epoch})
    print({'head': 'Train/AUC_head_1', 'auc': auc_extravasation.compute(), 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_2', 'acc': acc_kidney.compute(), 'epoch': epoch})
    print({'head': 'Train/AUC_head_2', 'auc': auc_kidney.compute(), 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_3', 'acc': acc_liver.compute(), 'epoch': epoch})
    print({'head': 'Train/AUC_head_3', 'auc': auc_liver.compute(), 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_4', 'acc': acc_spleen.compute(), 'epoch': epoch})
    print({'head': 'Train/AUC_head_4', 'auc': auc_spleen.compute(), 'epoch': epoch})

    if wandb is not None:
        run.log({
        'EPOCH': epoch,
        "bowel_acc": acc_bowel.compute(),
        "bowel_auc": auc_bowel.compute(),
        "extravasation_acc": acc_extravasation.compute(),
        "extravasation_auc": auc_extravasation.compute(),
        "kidney_acc": acc_kidney.compute(),
        "kidney_auc": auc_kidney.compute(),
        "liver_acc": acc_liver.compute(),
        "liver_auc": auc_liver.compute(),
        "spleen_acc": acc_spleen.compute(),
        "spleen_auc": auc_spleen.compute(),
        })

    # Validation
    model.eval()
    total_val_loss = 0.0

    # Metrics for Accuracy 
    val_acc_bowel = MulticlassAccuracy(num_classes=2)
    val_acc_extravasation = MulticlassAccuracy(num_classes=2)
    val_acc_kidney = MulticlassAccuracy(num_classes=3)
    val_acc_liver = MulticlassAccuracy(num_classes=3)
    val_acc_spleen = MulticlassAccuracy(num_classes=3)
    # Metrics for AUROC
    val_auc_bowel = MulticlassAUROC(num_classes=2)
    val_auc_extravasation = MulticlassAUROC(num_classes=2)
    val_auc_kidney = MulticlassAUROC(num_classes=3)
    val_auc_liver = MulticlassAUROC(num_classes=3)
    val_auc_spleen = MulticlassAUROC(num_classes=3)


    with torch.no_grad():
        for  i, (inputs, labels_dict) in enumerate(tqdm(val_loader)):
            # Move inputs to the device
            inputs = inputs.to(device)

            # Move labels in the dictionary to the device
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            labels_list = [labels_dict[key].to(device) for key in labels_dict]

            outputs = model(inputs)
            # Calculate each output loss
            loss1 = criterion1(outputs[0], labels_list[0][0])
            loss2 = criterion1(outputs[1], labels_list[1][0])
            loss3 = criterion2(outputs[2], labels_list[2])
            loss4 = criterion2(outputs[3], labels_list[3])
            loss5 = criterion2(outputs[4], labels_list[4])
            # Sum all the losses
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            total_val_loss += loss.item()

            # Calculate the accuracy
            val_acc_bowel.update(outputs[0].cpu(), labels_list[0][0].cpu())
            val_acc_extravasation.update(outputs[1].cpu(), labels_list[1][0].cpu())
            val_acc_kidney.update(outputs[2].cpu(), labels_list[2].cpu())
            val_acc_liver.update(outputs[3].cpu(), labels_list[3].cpu())
            val_acc_spleen.update(outputs[4].cpu(), labels_list[4].cpu())

            # Calculate the AUROC for each head
            val_auc_bowel.update(outputs[0].cpu(), labels_list[0][0].cpu())
            val_auc_extravasation.update(outputs[1].cpu(), labels_list[1][0].cpu())
            val_auc_kidney.update(outputs[2].cpu(), labels_list[2].cpu())
            val_auc_liver.update(outputs[3].cpu(), labels_list[3].cpu())
            val_auc_spleen.update(outputs[4].cpu(), labels_list[4].cpu())

    
    # Add to the history
    val_acc_history.append({"EPOCH": epoch,
                                "bowel": val_acc_bowel.compute(),
                                "extravasation": val_acc_extravasation.compute(),
                                "kidney": val_acc_kidney.compute(),
                                "liver": val_acc_liver.compute(),
                                "spleen": val_acc_spleen.compute()})
    val_auc_history.append({"EPOCH": epoch,
                                "bowel": val_auc_bowel.compute(),
                                "extravasation": val_auc_extravasation.compute(),
                                "kidney": val_auc_kidney.compute(),
                                "liver": val_auc_liver.compute(),
                                "spleen": val_auc_spleen.compute()})
    
    print({'EPOCH': epoch})
    print({'head': 'Validation/Accuracy_head_0', 'acc': val_acc_bowel.compute(), 'epoch': epoch})
    print({'head': 'Validation/AUC_head_0', 'auc': val_auc_bowel.compute(), 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_1', 'acc': val_acc_extravasation.compute(), 'epoch': epoch})
    print({'head': 'Validation/AUC_head_1', 'auc': val_auc_extravasation.compute(), 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_2', 'acc': val_acc_kidney.compute(), 'epoch': epoch})
    print({'head': 'Validation/AUC_head_2', 'auc': val_auc_kidney.compute(), 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_3', 'acc': val_acc_liver.compute(), 'epoch': epoch})
    print({'head': 'Validation/AUC_head_3', 'auc': val_auc_liver.compute(), 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_4', 'acc': val_acc_spleen.compute(), 'epoch': epoch})
    print({'head': 'Validation/AUC_head_4', 'auc': val_auc_spleen.compute(), 'epoch': epoch})

    if wandb is not None:
        run.log({
        'EPOCH': epoch,
        "val_bowel_acc": val_acc_bowel.compute(),
        "val_bowel_auc": val_auc_bowel.compute(),
        "val_extravasation_acc": val_acc_extravasation.compute(),
        "val_extravasation_auc": val_auc_extravasation.compute(),
        "val_kidney_acc": val_acc_kidney.compute(),
        "val_kidney_auc": val_auc_kidney.compute(),
        "val_liver_acc": val_acc_liver.compute(),
        "val_liver_auc": val_auc_liver.compute(),
        "val_spleen_acc": val_acc_spleen.compute(),
        "val_spleen_auc": val_auc_spleen.compute(),
        })



# Save the model
torch.save(model.state_dict(), f"{config.BASE_PATH}/model.pth")    

# Test the model
model.eval()
total_test_loss = 0.0
test_acc_history = []
test_auroc_history = []

test_acc_bowel = MulticlassAccuracy(num_classes=2)
test_acc_extravasation = MulticlassAccuracy(num_classes=2)
test_acc_kidney = MulticlassAccuracy(num_classes=3)
test_acc_liver = MulticlassAccuracy(num_classes=3)
test_acc_spleen = MulticlassAccuracy(num_classes=3)
test_auc_bowel = MulticlassAUROC(num_classes=2)
test_auc_extravasation = MulticlassAUROC(num_classes=2)
test_auc_kidney = MulticlassAUROC(num_classes=3)
test_auc_liver = MulticlassAUROC(num_classes=3)
test_auc_spleen = MulticlassAUROC(num_classes=3)


with torch.no_grad():
    for  i, (inputs, labels_dict) in enumerate(tqdm(test_loader)):
        # Move inputs to the device
        inputs = inputs.to(device)

        # Move labels in the dictionary to the device
        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
        labels_list = [labels_dict[key].to(device) for key in labels_dict]

        outputs = model(inputs)

        # No need to caluculate the loss for the test set
        
        # Calculate the accuracy
        test_acc_bowel.update(outputs[0].cpu(), labels_list[0][0].cpu())
        test_acc_extravasation.update(outputs[1].cpu(), labels_list[1][0].cpu())
        test_acc_kidney.update(outputs[2].cpu(), labels_list[2].cpu())
        test_acc_liver.update(outputs[3].cpu(), labels_list[3].cpu())
        test_acc_spleen.update(outputs[4].cpu(), labels_list[4].cpu())
        
        # Calculate the AUROC for each head
        test_auc_bowel.update(outputs[0].cpu(), labels_list[0][0].cpu())
        test_auc_extravasation.update(outputs[1].cpu(), labels_list[1][0].cpu())
        test_auc_kidney.update(outputs[2].cpu(), labels_list[2].cpu())
        test_auc_liver.update(outputs[3].cpu(), labels_list[3].cpu())
        test_auc_spleen.update(outputs[4].cpu(), labels_list[4].cpu())

print({'head': 'Test/Accuracy_head_0', 'acc': test_acc_bowel.compute()})
print({'head': 'Test/AUC_head_0', 'auc': test_auc_bowel.compute()})
print({'head': 'Test/Accuracy_head_1', 'acc': test_acc_extravasation.compute()})
print({'head': 'Test/AUC_head_1', 'auc': test_auc_extravasation.compute()})
print({'head': 'Test/Accuracy_head_2', 'acc': test_acc_kidney.compute()})
print({'head': 'Test/AUC_head_2', 'auc': test_auc_kidney.compute()})
print({'head': 'Test/Accuracy_head_3', 'acc': test_acc_liver.compute()})
print({'head': 'Test/AUC_head_3', 'auc': test_auc_liver.compute()})
print({'head': 'Test/Accuracy_head_4', 'acc': test_acc_spleen.compute()})
print({'head': 'Test/AUC_head_4', 'auc': test_auc_spleen.compute()})

if wandb is not None:
    run.log({
    "test_bowel_acc": test_acc_bowel.compute(),
    "test_bowel_auc": test_auc_bowel.compute(),
    "test_extravasation_acc": test_acc_extravasation.compute(),
    "test_extravasation_auc": test_auc_extravasation.compute(),
    "test_kidney_acc": test_acc_kidney.compute(),
    "test_kidney_auc": test_auc_kidney.compute(),
    "test_liver_acc": test_acc_liver.compute(),
    "test_liver_auc": test_auc_liver.compute(),
    "test_spleen_acc": test_acc_spleen.compute(),
    "test_spleen_auc": test_auc_spleen.compute(),
    })