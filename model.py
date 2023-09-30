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
    BASE_PATH = "/content/drive/MyDrive/kaggle/RSNA2023"

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

# dataframe['onelabel'] == 4882 is only one data and it cant split so conert it 4706
dataframe['onelabel'] = dataframe['onelabel'].apply(lambda x: 4706 if x == 4882 else x)

# Split the data
train_df, test_df = train_test_split(dataframe, test_size=0.5, random_state=42, stratify=dataframe['onelabel'])
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=43, stratify=test_df['onelabel'])
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

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
            'bowel': torch.argmax(torch.tensor(labels[:1], dtype=torch.float32)),
            'extravasation': torch.argmax(torch.tensor(labels[1:3], dtype=torch.float32)),
            'kidney': torch.argmax(torch.tensor(labels[3:6], dtype=torch.float32)),
            'liver': torch.argmax(torch.tensor(labels[6:9], dtype=torch.float32)),
            'spleen': torch.argmax(torch.tensor(labels[9:12], dtype=torch.float32)),
        }

        if self.transform:
            image = self.transform(image)

        return image, labels

# Define the transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize(256),
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
        precision = "fp16" # select either fp32 of fp16 (for better performance on GPU)
        self.base_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type=model_type, model_math=precision)

        # Freeze the parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the existing classifier with your own multi-head output
        out_features = 1000
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=out_features, bias=True)
        )
        

        # Define the new classifiers (heads)
        self.head1 = nn.Linear(out_features, 2)  # binary label
        self.head2 = nn.Linear(out_features, 2)  # binary label
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Train the model
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = [0] * 5

    for i, (inputs, labels_dict) in enumerate(tqdm(train_loader)):

        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
        labels_list = [labels_dict[key].to(device) for key in labels_dict]

        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = sum(criterion(output, label) for output, label in zip(outputs, labels_list))
        loss.backward()
        optimizer.step()

        #print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {loss.item()}")

        total_loss += loss.item()
        for i, output in enumerate(outputs):
            _, predicted = output.max(1)
            total_correct[i] += (predicted == labels_list[i]).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_accs = [correct / len(train_loader.dataset) for correct in total_correct]

    #writer.add_scalar('Train/Loss', avg_loss, epoch)
    print({'Train/Loss': avg_loss, 'EPOCH': epoch})
    # run.log({'Train/Loss': avg_loss, 'EPOCH': epoch})
    # for i, acc in enumerate(avg_accs):
    #     #writer.add_scalar(f'Train/Accuracy_head_{i}', acc, epoch)
    #     run.log({'head': f'Train/Accuracy_head_{i}',
    #               'acc':acc,
    #               'epoch':epoch})

    # Validation
    model.eval()
    total_val_loss = 0.0
    total_val_correct = [0] * 5

    with torch.no_grad():
        for  i, (inputs, labels_dict) in enumerate(tqdm(val_loader)):
            # Move inputs to the device
            inputs = inputs.to(device)

            # Move labels in the dictionary to the device
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            labels_list = [labels_dict[key].to(device) for key in labels_dict]

            outputs = model(inputs)
            loss = sum(criterion(output, label) for output, label in zip(outputs, labels_list))
            total_val_loss += loss.item()
            for i, output in enumerate(outputs):
                _, predicted = output.max(1)
                total_val_correct[i] += (predicted == labels_list[i]).sum().item()

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    avg_val_accs = [correct / len(val_loader.dataset) for correct in total_val_correct]

    #writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    print({'Validation/Loss': avg_val_loss, 'EPOCH': epoch})
    # run.log({'Validation/Loss': avg_val_loss, 'EPOCH': epoch})
    # for i, acc in enumerate(avg_val_accs):
    #     #writer.add_scalar(f'Validation/Accuracy_head_{i}', acc, epoch)
    #     run.log({'head': f'Validation/Accuracy_head_{i}',
    #               'acc':acc,
    #               'epoch':epoch})

# Save the model
torch.save(model.state_dict(), f"{config.BASE_PATH}/model.pth")

# Test the model
model.eval()
total_test_loss = 0.0
total_test_correct = [0] * 5

with torch.no_grad():
    for  i, (inputs, labels_dict) in enumerate(tqdm(test_loader)):
        # Move inputs to the device
        inputs = inputs.to(device)

        # Move labels in the dictionary to the device
        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
        labels_list = [labels_dict[key].to(device) for key in labels_dict]

        outputs = model(inputs)
        loss = sum(criterion(output, label) for output, label in zip(outputs, labels_list))
        total_test_loss += loss.item()
        for i, output in enumerate(outputs):
            _, predicted = output.max(1)
            total_test_correct[i] += (predicted == labels_list[i]).sum().item()

avg_test_loss = total_test_loss / len(test_loader.dataset)
avg_test_accs = [correct / len(test_loader.dataset) for correct in total_test_correct]

print({'Test/Loss': avg_test_loss})

# Save the entire model to a file
torch.save(model, 'model.pth')
