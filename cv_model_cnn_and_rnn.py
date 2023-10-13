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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


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
    FOCAL_LOSS_gamma = 2.5
    FOCAL_LOSS_alpha = 0.5
    config.SCHEDULER = "CosineAnnealingLR"  # CosineAnnealingLR or OneCycleLR

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
        "SCHEDULER":config.SCHEDULER,

        # focal loss
        "FOCAL_LOSS_gamma": config.FOCAL_LOSS_gamma,
        "FOCAL_LOSS_alpha": config.FOCAL_LOSS_alpha
}

# wandb setting
import wandb
mykey = '775553d8bb6bd0180350fae46e7d47bd9ef3f0d9'
if wandb is not None:
    wandb.login(key=mykey)

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

# Define the model
class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()

        # Load the pretrained efficientnet-b0
        #self.base_model = models.efficientnet_b0(pretrained=True)
        #self.base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

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

def multi_thresholding_with_original(thresholds=[0.3, 0.7]):
    def _multi_thresholding_with_original(img):
        # img should be a torch.Tensor with shape [1, H, W]
        channels = [img] + [(img > th).float() for th in thresholds]
        return torch.cat(channels, dim=0)  # Concatenate along the channel dimension
    return _multi_thresholding_with_original


# Define the transforms
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" (3 channels)
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Lambda(multi_thresholding_with_original([0.1, 0.9])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# To evaluate emsemble model, split the data into train and validation sets
dataframe.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True), test_df = train_test_split(dataframe, test_size=0.2, random_state=config.SEED, stratify=dataframe['stratify'])

# Stratified k-fold cross validation
k_folds = 3
labels = dataframe['stratify'].values
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.SEED)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_models = []  # Store the models for each fold

for fold, (train_ids, val_ids) in enumerate(skf.split(dataframe, labels)):
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Split the data into train and validation sets
    train_df = dataframe.iloc[train_ids]
    val_df = dataframe.iloc[val_ids]

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

    # wandb setting
    import wandb
    mykey = '775553d8bb6bd0180350fae46e7d47bd9ef3f0d9'
    # wandb = None
    if wandb is not None:
        run = wandb.init(project="RSNA2023", config=wandb_config)

    # Define the model
    model = CNNRNN()
    model = model.to(device)

    # Define the loss function and optimizer
    #criterion1 = nn.BCEWithLogitsLoss()
    criterion1 = FocalLoss(alpha=config.FOCAL_LOSS_alpha, gamma=config.FOCAL_LOSS_gamma)
    #criterion2 = nn.CrossEntropyLoss()
    criterion2 = MultiClassFocalLoss(alpha=config.FOCAL_LOSS_alpha, gamma=config.FOCAL_LOSS_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    if config.SCHEDULER == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)

    # Train the model
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        # Metrics for Accuracy
        acc_bowel = BinaryAccuracy()
        acc_extravasation =BinaryAccuracy()
        acc_kidney = MulticlassAccuracy(num_classes=3)
        acc_liver = MulticlassAccuracy(num_classes=3)
        acc_spleen = MulticlassAccuracy(num_classes=3)
        # Metrics for AUROC
        auc_bowel = BinaryAUROC()
        auc_extravasation = BinaryAUROC()
        auc_kidney = MulticlassAUROC(num_classes=3)
        auc_liver = MulticlassAUROC(num_classes=3)
        auc_spleen = MulticlassAUROC(num_classes=3)

        acc_all = BinaryAccuracy()
        auc_all = BinaryAUROC()

        accs = [acc_bowel, acc_extravasation, acc_kidney, acc_liver, acc_spleen, acc_all]
        aucs = [auc_bowel, auc_extravasation, auc_kidney, auc_liver, auc_spleen, auc_all]

        for i, (inputs, lengths, labels_dict) in enumerate(tqdm(train_loader)):

            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            labels_list = [labels_dict[key].to(device) for key in labels_dict]

            optimizer.zero_grad()
            inputs = inputs.to(device)
            outputs = model(inputs, lengths)
            outputs = [output.squeeze(-1).to(device) if output.shape[-1] == 1 else output.to(device) for output in outputs]

            # Calculate each output loss
            losses = [criterion1(outputs[i], labels_list[i].float()) if i in [0, 1, 5] else criterion2(outputs[i], labels_list[i]) for i in range(6)]
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Get the probabilities and labels for each head
            #probs = [torch.sigmoid(outputs[0]).cpu(), torch.sigmoid(outputs[1]).cpu(), F.softmax(outputs[2], dim=1).cpu(), F.softmax(outputs[3], dim=1).cpu(), F.softmax(outputs[4], dim=1).cpu(), torch.sigmoid(outputs[5]).cpu()]
            probs = [torch.sigmoid(outputs[i]).cpu() if i in [0, 1, 5] else F.softmax(outputs[i], dim=1).cpu() for i in range(6)]
            labels = [labels_list[0].cpu(), labels_list[1].cpu(), labels_list[2].cpu(), labels_list[3].cpu(), labels_list[4].cpu(), labels_list[5].cpu()]

            # Update the accuracy and AUROC metrics for each head
            for i in range(6):
                accs[i].update(probs[i], labels[i])
                aucs[i].update(probs[i], labels[i])

        # Compute the epoch acc and auc for each head
        res_accs = [accs[i].compute() for i in range(6)]
        res_aucs = [aucs[i].compute() for i in range(6)]

        # Print the resultsS
        print({'EPOCH': epoch})
        for i in range(6):
            print({'head': f'Train/Accuracy_head_{i}', 'acc': res_accs[i], 'epoch': epoch})
            print({'head': f'Train/AUC_head_{i}', 'auc': res_aucs[i], 'epoch': epoch})

        # Log the results to WandB
        if wandb is not None:
            run.log({
                'EPOCH': epoch,
                "train_bowel_acc": res_accs[0],
                "train_bowel_auc": res_aucs[0],
                "train_extravasation_acc": res_accs[1],
                "train_extravasation_auc": res_aucs[1],
                "train_kidney_acc": res_accs[2],
                "train_kidney_auc": res_aucs[2],
                "train_liver_acc": res_accs[3],
                "train_liver_auc": res_aucs[3],
                "train_spleen_acc": res_accs[4],
                "train_spleen_auc": res_aucs[4],
                "train_all_acc": res_accs[5],
                "train_all_auc": res_aucs[5],
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

        val_accs = [val_acc_bowel, val_acc_extravasation, val_acc_kidney, val_acc_liver, val_acc_spleen, val_acc_all]
        val_aucs = [val_auc_bowel, val_auc_extravasation, val_auc_kidney, val_auc_liver, val_auc_spleen, val_auc_all]


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
                losses = [criterion1(outputs[i], labels_list[i].float()) if i in [0, 1, 5] else criterion2(outputs[i], labels_list[i]) for i in range(6)]
                loss = sum(losses)
                total_val_loss += loss.item()

                # Get the probabilities
                probs = [torch.sigmoid(outputs[i]).cpu() if i in [0, 1, 5] else F.softmax(outputs[i], dim=1).cpu() for i in range(6)]
                labels = [labels_list[0].cpu(), labels_list[1].cpu(), labels_list[2].cpu(), labels_list[3].cpu(), labels_list[4].cpu(), labels_list[5].cpu()]
                
                # Calculate the accuracy and the AUROC for each head
                for i in range(6):
                    val_accs[i].update(probs[i], labels[i])
                    val_aucs[i].update(probs[i], labels[i])

        # Compute the epoch acc and auc
        for i in range(6):
            val_accs[i] = val_accs[i].compute()
            val_aucs[i] = val_aucs[i].compute()

        # Print the results
        print({'EPOCH': epoch})
        for i in range(6):
            print({'head': f'Validation/Accuracy_head_{i}', 'acc': val_accs[i], 'epoch': epoch})
            print({'head': f'Validation/AUC_head_{i}', 'auc': val_aucs[i], 'epoch': epoch})
        
        # Log the results to WandB
        if wandb is not None:
            run.log({
                'EPOCH': epoch,
                "val_bowel_acc": val_accs[0],
                "val_bowel_auc": val_aucs[0],
                "val_extravasation_acc": val_accs[1],
                "val_extravasation_auc": val_aucs[1],
                "val_kidney_acc": val_accs[2],
                "val_kidney_auc": val_aucs[2],
                "val_liver_acc": val_accs[3],
                "val_liver_auc": val_aucs[3],
                "val_spleen_acc": val_accs[4],
                "val_spleen_auc": val_aucs[4],
                "val_all_acc": val_accs[5],
                "val_all_auc": val_aucs[5],
            })
        
    # Save the model as JIT and move it to CPU
    model.eval()
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(f'/content/drive/MyDrive/kaggle/RSNA2023/models/cv_convnext_{fold}.pt') # Save

    model.to('cpu')
    train_models.append(model)

    # finish the wandb run
    if wandb is not None:
        run.finish()

    # Save the model
    # torch.save(model.state_dict(), f"{config.BASE_PATH}/cnn_rnn_model3.pth")



# Test the model
def post_proc(pred):
    proc_pred = np.empty((pred.shape[0], 2*2 + 3*3), dtype="float32")

    # bowel, extravasation
    proc_pred[:, 0] = 1 - pred[:, 0]
    proc_pred[:, 1] = pred[:, 0]
    proc_pred[:, 2] = 1 - pred[:, 1]
    proc_pred[:, 3] = pred[:, 1]
    
    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[:, 2:5]
    proc_pred[:, 7:10] = pred[:, 5:8]
    proc_pred[:, 10:13] = pred[:, 8:11]

    return proc_pred

# Define the test dataset

test_df["length"] = test_df["image_path"].apply(lambda x: len(x))
test_df = test_df.sort_values(by="length", ascending=False, ignore_index=True)
test_dataset = CT_Slices_Dataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

# Test the model
def post_proc(pred):
    proc_pred = np.empty((pred.shape[0], 2*2 + 3*3), dtype="float32")

    # bowel, extravasation
    proc_pred[:, 0] = 1 - pred[:, 0]
    proc_pred[:, 1] = pred[:, 0]
    proc_pred[:, 2] = 1 - pred[:, 1]
    proc_pred[:, 3] = pred[:, 1]
    
    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[:, 2:5]
    proc_pred[:, 7:10] = pred[:, 5:8]
    proc_pred[:, 10:13] = pred[:, 8:11]

    return proc_pred

# Initializing array to store predictions
patient_preds = np.zeros(
    shape=(len(test_df.patient_id), 2*2 + 3*3),
    dtype="float32"
)

# Lists to store AUC scores for each fold
auc_bowel = []
auc_extravasation = []
auc_kidney = []
auc_liver = []
auc_spleen = []

with torch.no_grad():
    for fold in range(k_folds):
        print(f"FOLD {fold}")
        print("--------------------------------")
        # Assuming each batch produces outputs of shape (batch_size, feature_dim)
        current_index = 0

        # Initializing array to store predictions
        fold_patient_preds = np.zeros(
            shape=(len(test_df.patient_id), 2*2 + 3*3),
            dtype="float32"
        )
        model = torch.jit.load(f'/content/drive/MyDrive/kaggle/RSNA2023/models/cv_model_{fold}.pt', map_location=torch.device(device))
        model = model.to(device)
        for i, (inputs, lengths, _) in enumerate(tqdm(test_loader)):
            # Move inputs to the device
            inputs = inputs.to(device)
            outputs = model(inputs, lengths)
            
            # Apply softmax to the outputs (except for the first two which use BCELoss)
            probs_bowel = torch.sigmoid(outputs[0]).cpu()
            probs_extravasation = torch.sigmoid(outputs[1]).cpu()
            probs_kidney = F.softmax(outputs[2], dim=1).cpu()
            probs_liver = F.softmax(outputs[3], dim=1).cpu()
            probs_spleen = F.softmax(outputs[4], dim=1).cpu()
            
            # multi-headの出力を1行に結合します
            predictions_reshaped = torch.cat([probs_bowel, probs_extravasation, probs_kidney, probs_liver, probs_spleen], dim=1)         
            predictions_reshaped = post_proc(predictions_reshaped)

            for i in range(len(predictions_reshaped)):
                patient_preds[current_index+i] += predictions_reshaped[i]
                fold_patient_preds[current_index+i] += predictions_reshaped[i]
            current_index += config.BATCH_SIZE

        pred_df = pd.DataFrame({"patient_id":test_df["patient_id"],})
        pred_df[config.TARGET_COLS[:-1]] = fold_patient_preds # except for 'all' column

        test_auc_bowel = roc_auc_score(test_df.loc[:, "bowel_injury"].values, pred_df.loc[:, "bowel_injury"].values)
        test_auc_extravasation = roc_auc_score(test_df.loc[:, "extravasation_injury"].values, pred_df.loc[:, "extravasation_injury"].values)
        test_auc_kidney = roc_auc_score(test_df.loc[:, ["kidney_healthy", "kidney_high"]].values, pred_df.loc[:, ["kidney_healthy", "kidney_high"]].values, multi_class="ovr")
        test_auc_liver = roc_auc_score(test_df.loc[:, ["liver_healthy", "liver_high"]].values, pred_df.loc[:, ["liver_healthy", "liver_high"]].values, multi_class="ovr")
        test_auc_spleen = roc_auc_score(test_df.loc[:, ["spleen_healthy", "spleen_high"]].values, pred_df.loc[:, ["spleen_healthy", "spleen_high"]].values, multi_class="ovr")
        
        auc_bowel.append(test_auc_bowel)
        auc_extravasation.append(test_auc_extravasation)
        auc_kidney.append(test_auc_kidney)
        auc_liver.append(test_auc_liver)
        auc_spleen.append(test_auc_spleen)

# 各列の平均を取る
patient_preds = patient_preds / k_folds

# Create Submission
pred_df = pd.DataFrame({"patient_id":val_df["patient_id"],})
pred_df[config.TARGET_COLS[:-1]] = patient_preds # eccept for 'all' column

# Caluculate the score with sklearn metrics


test_auc_bowel = roc_auc_score(test_df.loc[:, "bowel_injury"].values, pred_df.loc[:, "bowel_injury"].values)
test_auc_extravasation = roc_auc_score(test_df.loc[:, "extravasation_injury"].values, pred_df.loc[:, "extravasation_injury"].values)
test_auc_kidney = roc_auc_score(test_df.loc[:, ["kidney_healthy", "kidney_high"]].values, pred_df.loc[:, ["kidney_healthy", "kidney_high"]].values, multi_class="ovr")
test_auc_liver = roc_auc_score(test_df.loc[:, ["liver_healthy", "liver_high"]].values, pred_df.loc[:, ["liver_healthy", "liver_high"]].values, multi_class="ovr")
test_auc_spleen = roc_auc_score(test_df.loc[:, ["spleen_healthy", "spleen_high"]].values, pred_df.loc[:, ["spleen_healthy", "spleen_high"]].values, multi_class="ovr")


# Print the results
print({'head': f'Test/AUC_bowel', 'auc': test_auc_bowel})
print({'head': f'Test/AUC_extravasation', 'auc': test_auc_extravasation})
print({'head': f'Test/AUC_kidney', 'auc': test_auc_kidney})
print({'head': f'Test/AUC_liver', 'auc': test_auc_liver})
print({'head': f'Test/AUC_spleen', 'auc': test_auc_spleen})

# Compute variance for each list
var_auc_bowel = np.var(auc_bowel)
var_auc_extravasation = np.var(auc_extravasation)
var_auc_kidney = np.var(auc_kidney)
var_auc_liver = np.var(auc_liver)
var_auc_spleen = np.var(auc_spleen)

print("Variance of AUC for bowel:", var_auc_bowel)
print("Variance of AUC for extravasation:", var_auc_extravasation)
print("Variance of AUC for kidney:", var_auc_kidney)
print("Variance of AUC for liver:", var_auc_liver)
print("Variance of AUC for spleen:", var_auc_spleen)

# wandb = None
if wandb is not None:
    run = wandb.init(project="RSNA2023", config=wandb_config)
    run.log({
        "test_bowel_auc": test_auc_bowel,
        "test_extravasation_auc": test_auc_extravasation,
        "test_kidney_auc": test_auc_kidney,
        "test_liver_auc": test_auc_liver,
        "test_spleen_auc": test_auc_spleen,
        "var_auc_bowel": var_auc_bowel,
        "var_auc_extravasation": var_auc_extravasation,
        "var_auc_kidney": var_auc_kidney,
        "var_auc_liver": var_auc_liver,
        "var_auc_spleen": var_auc_spleen,
    })
    run.finish()