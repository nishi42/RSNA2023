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
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, BinaryAccuracy, BinaryAUROC
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.restoration import denoise_nl_means, estimate_sigma
from torchvision.transforms import ToPILImage, ToTensor

# Set up the configuration
class Config:
    SEED = 42
    IMAGE_SIZE = [512, 512]
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    TARGET_COLS  = [
        "bowel_healthy","bowel_injury", "extravasation_healthy","extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high", 
        "all"
    ]
    BASE_PATH = "/content/drive/MyDrive/kaggle/RSNA2023"
    HALF = False
    MODEL = "ConvNeXt_Tiny and bi-LSTM(layer=1)"
    DENOISE = True
    CT_STACK_SIZE = 16
    MAX_SEQ_LEN = 50
    CROP = None
    FREEZE = 6

# Set up the configuration
config = Config()
wandb_config = {
        "SEED":config.SEED,
        "IMAGE_SIZE":config.IMAGE_SIZE,
        "HALF":config.HALF,
        "DENOISE":config.DENOISE,
        "MODEL":config.MODEL,
        "BATCH_SIZE":config.BATCH_SIZE,
        "MAX_SEQ_LEN":config.MAX_SEQ_LEN,
        "EPOCHS":config.EPOCHS,
        "FREEZE":config.FREEZE,

        "SCHEDULER":"CosineAnnealingLR",
        "FOCAL_LOSS_gamma": 2.5,
        "FOCAL_LOSS_alpha": 0.5
}

# Load the data
dataframe = pd.read_csv(f"{config.BASE_PATH}/train.csv")
dataframe["image_path"] = f"{config.BASE_PATH}/train_images"\
                    + "/" + dataframe.patient_id.astype(str)\
                    + "/" + dataframe.series_id.astype(str)\
                    + "/" + dataframe.instance_number.astype(str) +".png"
dataframe = dataframe.drop_duplicates()

# group by the patient_id, labels and concat image path list
dataframe = dataframe.groupby(['patient_id', 'series_id', 'bowel_healthy', 'bowel_injury', 'extravasation_healthy', 'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high'])['image_path'].apply(list).reset_index(name='image_path')

# choose only lateset CTs
# Get the index of the rows with the maximum series_id for each patient_id
idx = dataframe.groupby('patient_id')['series_id'].idxmax()

# Use the index to filter the dataframe
dataframe = dataframe.loc[idx]

# Stratify by the spleen and kindey (Those are not good learned by the model)
dataframe['stratify'] = ''
for col in config.TARGET_COLS:
    if col in [ "bowel_healthy","bowel_injury", "extravasation_healthy","extravasation_injury"]:
        dataframe['stratify'] += dataframe[col].astype(str)

# Check all healthy columns are 0.
# If all healthy columns are 1, then 'all' columns is 0
# If one of the healthy columns is 0, then 'all' columns is 1(this means the patient has at least one injury)
dataframe['all'] = 0
healthy_cols = ['bowel_healthy', 'extravasation_healthy', 'kidney_healthy', 'liver_healthy', 'spleen_healthy']
for col in healthy_cols:
    dataframe['all'] += dataframe[col]
dataframe['all'] = dataframe['all'].apply(lambda x: 0 if x == 5 else 1)

# Define the dataset
class CT_Slices_Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    # check the image path list length and uniform sampling
    def uniform_temporal_subsample(self, ima_file_list, sample_size):
        step_size = len(ima_file_list) // sample_size
        downsampled_list = ima_file_list[::step_size][:sample_size]
        return downsampled_list

    def weighted_temporal_subsample(self, ima_file_list, sample_size):
        # Create a probability distribution
        total_slices = len(ima_file_list)
        slice_indices = np.arange(total_slices)
        center_idx = total_slices // 2

        # Gaussian distribution centered around the middle slice
        probabilities = np.exp(-((slice_indices - center_idx) ** 2) / (2 * (total_slices * 0.1) ** 2))
        probabilities /= probabilities.sum()  # Normalize to sum to 1

        # Sample using the created distribution
        selected_indices = np.random.choice(slice_indices, size=sample_size, replace=False, p=probabilities)
        selected_indices = sorted(selected_indices)  # Sort the indices for sequential access

        return [ima_file_list[i] for i in selected_indices]


    def __getitem__(self, idx):
        ima_file_list = self.dataframe.loc[idx, 'image_path']

        # uniform sampling
        downsample_size = config.MAX_SEQ_LEN
        if len(ima_file_list) > downsample_size:
            ima_file_list = self.uniform_temporal_subsample(ima_file_list, downsample_size)
            #ima_file_list = self.weighted_temporal_subsample(ima_file_list, downsample_size)

        slices = []
        for slice_file in sorted(ima_file_list):
            slice_img = Image.open(slice_file)
            if self.transform:
                slice_img = self.transform(slice_img)
            slices.append(slice_img)

        slices = torch.stack(slices)

        labels = self.dataframe.loc[idx, config.TARGET_COLS]
        labels = {
            'bowel': torch.argmax(torch.tensor(labels[:2], dtype=torch.float32)), # binary label, [0, 1]
            'extravasation': torch.argmax(torch.tensor(labels[2:4], dtype=torch.float32)), # binary label, [0, 1, 2]
            'kidney': torch.argmax(torch.tensor(labels[4:7], dtype=torch.float32)), # multi-class label, [0, 1, 2]
            'liver': torch.argmax(torch.tensor(labels[7:10], dtype=torch.float32)), # multi-class label, [0, 1, 2]
            'spleen': torch.argmax(torch.tensor(labels[10:13], dtype=torch.float32)), # multi-class label, [0, 1, 2]
            'all': torch.argmax(torch.tensor(labels[13], dtype=torch.float32)), # binary label, [0, 1]
        }

        return slices, len(slices), labels

# collate_fn for DataLoader with applying padding
def collate_fn(batch):
    sequences, lengths, labels_list = zip(*batch)

    # padding
    sequences_padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor(lengths)

    # Combine the labels from all samples in the batch into a single dictionary
    keys = labels_list[0].keys()
    labels_dict = {key: torch.stack([d[key] for d in labels_list]) for key in keys}

    return sequences_padded, lengths, labels_dict

# Define the transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize([256, 256]),
    transforms.CenterCrop([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split the data into train and validation sets
train_df, val_df = train_test_split(dataframe,
                                    test_size=0.2,
                                    random_state=config.SEED,
                                    shuffle=True,
                                    stratify=dataframe['stratify'])

# Split the train data into train and validation sets
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)

# Sort the length of image data path
train_df["length"] = train_df["image_path"].apply(lambda x: len(x))
train_df = train_df.sort_values(by="length", ascending=False, ignore_index=True)
val_df["length"] = val_df["image_path"].apply(lambda x: len(x))
val_df = val_df.sort_values(by="length", ascending=False, ignore_index=True)

# Create the train and validation datasets

train_dataset = CT_Slices_Dataset(train_df, transform=transform)
val_dataset = CT_Slices_Dataset(val_df, transform=transform)
# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

# Define the model
class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()

        # Load the pretrained efficientnet-b0
        #self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # # Freeze the parameters
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Freeze only the first few layers
        for param in list(self.base_model.parameters())[:-config.FREEZE]:
            param.requires_grad = False

        # CNN output features
        cnn_out_features = 1000

        # RNN output features
        output_features = 256

        # Fully connected layer output features
        fc_out_features = 256

        # RNN (LSTM) layers
        self.rnn = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=output_features,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Linear(2 * output_features, fc_out_features)

        # Batch Normalization layer
        self.bn = nn.BatchNorm1d(fc_out_features)

        # Drop out layer
        self.dropout = nn.Dropout(0.5)   

        # Define the new classifiers (heads)
        self.head1 = nn.Linear(fc_out_features, 1)  # binary label
        self.head2 = nn.Linear(fc_out_features, 1)  # binary label
        self.head3 = nn.Linear(fc_out_features, 3)  # multi-class label
        self.head4 = nn.Linear(fc_out_features, 3)  # multi-class label
        self.head5 = nn.Linear(fc_out_features, 3)  # multi-class label
        self.head6 = nn.Linear(fc_out_features, 1)  # multi-class label

    def forward(self, x, lengths):
        batch_size, timesteps, C, H, W = x.size()
        cnn_out = self.base_model(x.view(batch_size * timesteps, C, H, W))
        cnn_out = cnn_out.view(batch_size, timesteps, -1)

        # pack_padded_sequenceを使用
        packed_input = pack_padded_sequence(cnn_out, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(rnn_out[:, -1, :])
        out = self.bn(out)
        out = self.dropout(out)

        out1 = self.head1(out)
        out2 = self.head2(out)
        out3 = self.head3(out)
        out4 = self.head4(out)
        out5 = self.head5(out)
        out6 = self.head6(out)

        return out1, out2, out3, out4, out5, out6

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.5, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, pt = 1-p if y=0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.5, reduction='sum'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # ソフトマックスを使用して確率を計算
        probs = F.softmax(inputs, dim=1)
        
        # 選択されたクラスの確率を取得
        class_probs = probs.gather(1, targets.view(-1, 1))
        
        # Focal Lossの計算
        F_loss = -self.alpha * (1-class_probs)**self.gamma * class_probs.log()
        
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
model = CNNRNN()
model = model.to(device)

# Define the loss function and optimizer
#criterion1 = nn.BCEWithLogitsLoss()
criterion1 = FocalLoss(alpha=0.5, gamma=3)
#criterion2 = nn.CrossEntropyLoss()
criterion2 = MultiClassFocalLoss(alpha=0.5, gamma=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#scheduler = CosineAnnealingLR(optimizer, T_max=10)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)

# wandb setting
import wandb
mykey = '775553d8bb6bd0180350fae46e7d47bd9ef3f0d9'
# wandb = None
if wandb is not None:
    wandb.login(key=mykey)
    run = wandb.init(project="RSNA2023", config=wandb_config)

# Train the model
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0.0

    # Metrics for Accuracy and AUROC
    acc_bowel = BinaryAccuracy()
    acc_extravasation =BinaryAccuracy()
    acc_kidney = MulticlassAccuracy(num_classes=3)
    acc_liver = MulticlassAccuracy(num_classes=3)
    acc_spleen = MulticlassAccuracy(num_classes=3)
    auc_bowel = BinaryAUROC()
    auc_extravasation = BinaryAUROC()
    auc_kidney = MulticlassAUROC(num_classes=3)
    auc_liver = MulticlassAUROC(num_classes=3)
    auc_spleen = MulticlassAUROC(num_classes=3)

    acc_all = BinaryAccuracy()
    auc_all = BinaryAUROC()

    for i, (inputs, lengths, labels_dict) in enumerate(tqdm(train_loader)):

        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
        labels_list = [labels_dict[key].to(device) for key in labels_dict]

        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs, lengths)
        outputs = [output.squeeze(-1).to(device) if output.shape[-1] == 1 else output.to(device) for output in outputs]

        # Calculate each output loss
        loss1 = criterion1(outputs[0], labels_list[0].float())
        loss2 = criterion1(outputs[1], labels_list[1].float())
        loss3 = criterion2(outputs[2], labels_list[2])
        loss4 = criterion2(outputs[3], labels_list[3])
        loss5 = criterion2(outputs[4], labels_list[4])
        loss6 = criterion1(outputs[5], labels_list[5].float())
        # Sum all the losses
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Get the probabilities
        probs_bowel = torch.sigmoid(outputs[0]).cpu()
        probs_extravasation = torch.sigmoid(outputs[1]).cpu()
        probs_kidney = F.softmax(outputs[2], dim=1).cpu()
        probs_liver = F.softmax(outputs[3], dim=1).cpu()
        probs_spleen = F.softmax(outputs[4], dim=1).cpu()
        probs_all = torch.sigmoid(outputs[5]).cpu()

        # Calculate the accuracy
        acc_bowel.update(probs_bowel, labels_list[0].cpu())
        acc_extravasation.update(probs_extravasation, labels_list[1].cpu())
        acc_kidney.update(probs_kidney, labels_list[2].cpu())
        acc_liver.update(probs_liver, labels_list[3].cpu())
        acc_spleen.update(probs_spleen, labels_list[4].cpu())
        acc_all.update(probs_all, labels_list[5].cpu())

        # Calculate the AUROC for each head
        auc_bowel.update(probs_bowel, labels_list[0].cpu())
        auc_extravasation.update(probs_extravasation, labels_list[1].cpu())
        auc_kidney.update(probs_kidney, labels_list[2].cpu())
        auc_liver.update(probs_liver, labels_list[3].cpu())
        auc_spleen.update(probs_spleen, labels_list[4].cpu())
        auc_all.update(probs_all, labels_list[5].cpu())

    # Compute the epoch acc and auc
    res_acc_bowel = acc_bowel.compute()
    res_acc_extravasation = acc_extravasation.compute()
    res_acc_kidney = acc_kidney.compute()
    res_acc_liver = acc_liver.compute()
    res_acc_spleen = acc_spleen.compute()
    res_acc_all = acc_all.compute()

    res_auc_bowel = auc_bowel.compute()
    res_auc_extravasation = auc_extravasation.compute()
    res_auc_kidney = auc_kidney.compute()
    res_auc_liver = auc_liver.compute()
    res_auc_spleen = auc_spleen.compute()
    res_auc_all = auc_all.compute()

    print({'EPOCH': epoch})
    print({'head': 'Train/Accuracy_head_0', 'acc': res_acc_bowel, 'epoch': epoch})
    print({'head': 'Train/AUC_head_0', 'auc': res_auc_bowel, 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_1', 'acc': res_acc_extravasation, 'epoch': epoch})
    print({'head': 'Train/AUC_head_1', 'auc': res_auc_extravasation, 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_2', 'acc': res_acc_kidney, 'epoch': epoch})
    print({'head': 'Train/AUC_head_2', 'auc': res_auc_kidney, 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_3', 'acc': res_acc_liver, 'epoch': epoch})
    print({'head': 'Train/AUC_head_3', 'auc': res_auc_liver, 'epoch': epoch})
    print({'head': 'Train/Accuracy_head_4', 'acc': res_acc_spleen, 'epoch': epoch})
    print({'head': 'Train/AUC_head_4', 'auc': res_auc_spleen, 'epoch': epoch})

    if wandb is not None:
        run.log({
        'EPOCH': epoch,
        "train_bowel_acc": res_acc_bowel,
        "train_bowel_auc": res_auc_bowel,
        "train_extravasation_acc": res_acc_extravasation,
        "train_extravasation_auc": res_auc_extravasation,
        "train_kidney_acc": res_acc_kidney,
        "train_kidney_auc": res_auc_kidney,
        "train_liver_acc": res_acc_liver,
        "train_liver_auc": res_auc_liver,
        "train_spleen_acc": res_acc_spleen,
        "train_spleen_auc": res_auc_spleen,
        "train_all_acc": res_acc_all,
        "train_all_auc": res_auc_all,
        })

    # Validation
    model.eval()
    total_val_loss = 0.0

    # Metrics for Accuracy
    val_acc_bowel = BinaryAccuracy()
    val_acc_extravasation = BinaryAccuracy()
    val_acc_kidney = MulticlassAccuracy(num_classes=3)
    val_acc_liver = MulticlassAccuracy(num_classes=3)
    val_acc_spleen = MulticlassAccuracy(num_classes=3)
    # Metrics for AUROC
    val_auc_bowel = BinaryAUROC()
    val_auc_extravasation = BinaryAUROC()
    val_auc_kidney = MulticlassAUROC(num_classes=3)
    val_auc_liver = MulticlassAUROC(num_classes=3)
    val_auc_spleen = MulticlassAUROC(num_classes=3)

    val_acc_all = BinaryAccuracy()
    val_auc_all = BinaryAUROC()


    with torch.no_grad():
        for  i, (inputs, lengths, labels_dict) in enumerate(tqdm(val_loader)):
            # Move inputs to the device
            inputs = inputs.to(device)

            # Move labels in the dictionary to the device
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            labels_list = [labels_dict[key].to(device) for key in labels_dict]

            outputs = model(inputs, lengths)
            outputs = [output.squeeze(-1).to(device) if output.shape[-1] == 1 else output.to(device) for output in outputs]
            # Calculate each output loss
            loss1 = criterion1(outputs[0], labels_list[0].float())
            loss2 = criterion1(outputs[1], labels_list[1].float())
            loss3 = criterion2(outputs[2], labels_list[2])
            loss4 = criterion2(outputs[3], labels_list[3])
            loss5 = criterion2(outputs[4], labels_list[4])
            loss6 = criterion1(outputs[5], labels_list[5].float())
            # Sum all the losses
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            total_val_loss += loss.item()

            # Get the probabilities
            probs_bowel = torch.sigmoid(outputs[0]).cpu()
            probs_extravasation = torch.sigmoid(outputs[1]).cpu()
            probs_kidney = F.softmax(outputs[2], dim=1).cpu()
            probs_liver = F.softmax(outputs[3], dim=1).cpu()
            probs_spleen = F.softmax(outputs[4], dim=1).cpu()
            probs_all = torch.sigmoid(outputs[5]).cpu()

            # Calculate the accuracy
            val_acc_bowel.update(probs_bowel, labels_list[0].cpu())
            val_acc_extravasation.update(probs_extravasation, labels_list[1].cpu())
            val_acc_kidney.update(probs_kidney, labels_list[2].cpu())
            val_acc_liver.update(probs_liver, labels_list[3].cpu())
            val_acc_spleen.update(probs_spleen, labels_list[4].cpu())
            val_acc_all.update(probs_all, labels_list[5].cpu())

            # Calculate the AUROC for each head
            val_auc_bowel.update(probs_bowel, labels_list[0].cpu())
            val_auc_extravasation.update(probs_extravasation, labels_list[1].cpu())
            val_auc_kidney.update(probs_kidney, labels_list[2].cpu())
            val_auc_liver.update(probs_liver, labels_list[3].cpu())
            val_auc_spleen.update(probs_spleen, labels_list[4].cpu())
            val_auc_all.update(probs_all, labels_list[5].cpu())

    # Compute the epoch acc and auc
    res_val_acc_bowel = val_acc_bowel.compute()
    res_val_acc_extravasation = val_acc_extravasation.compute()
    res_val_acc_kidney = val_acc_kidney.compute()
    res_val_acc_liver = val_acc_liver.compute()
    res_val_acc_spleen = val_acc_spleen.compute()
    res_val_acc_all = val_acc_all.compute()

    res_val_auc_bowel = val_auc_bowel.compute()
    res_val_auc_extravasation = val_auc_extravasation.compute()
    res_val_auc_kidney = val_auc_kidney.compute()
    res_val_auc_liver = val_auc_liver.compute()
    res_val_auc_spleen = val_auc_spleen.compute()
    res_val_auc_all = val_auc_all.compute()

    print({'EPOCH': epoch})
    print({'head': 'Validation/Accuracy_head_0', 'acc': res_val_acc_bowel, 'epoch': epoch})
    print({'head': 'Validation/AUC_head_0', 'auc': res_val_auc_bowel, 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_1', 'acc': res_val_acc_extravasation, 'epoch': epoch})
    print({'head': 'Validation/AUC_head_1', 'auc': res_val_auc_extravasation, 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_2', 'acc': res_val_acc_kidney, 'epoch': epoch})
    print({'head': 'Validation/AUC_head_2', 'auc': res_val_auc_kidney, 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_3', 'acc': res_val_acc_liver, 'epoch': epoch})
    print({'head': 'Validation/AUC_head_3', 'auc': res_val_auc_liver, 'epoch': epoch})
    print({'head': 'Validation/Accuracy_head_4', 'acc': res_val_acc_spleen, 'epoch': epoch})
    print({'head': 'Validation/AUC_head_4', 'auc': res_val_auc_spleen, 'epoch': epoch})


    if wandb is not None:
        run.log({
        'EPOCH': epoch,
        "val_bowel_acc": res_val_acc_bowel,
        "val_bowel_auc": res_val_auc_bowel,
        "val_extravasation_acc": res_val_acc_extravasation,
        "val_extravasation_auc": res_val_auc_extravasation,
        "val_kidney_acc": res_val_acc_kidney,
        "val_kidney_auc": res_val_auc_kidney,
        "val_liver_acc": res_val_acc_liver,
        "val_liver_auc": res_val_auc_liver,
        "val_spleen_acc": res_val_acc_spleen,
        "val_spleen_auc": res_val_auc_spleen,
        "val_all_acc": res_val_acc_all,
        "val_all_auc": res_val_auc_all,
        })

# Save the model
# torch.save(model.state_dict(), f"{config.BASE_PATH}/cnn_rnn_model3.pth")