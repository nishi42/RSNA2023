import numpy as np
import torch
import torcheval
import pandas as pd

# Convert the label to the original label
def convert_label(label):
    # Convert the label to the original label
    label = int(label)
    # Convert the label to the original label
    label = label_dict[label]
    return label

# Convert binary string to list of digits
def convert_binary_str_to_list(binary_str):
    digits = np.array([])
    for char in binary_str:
        digits.append(int(char))
    return digits

label_dict = {
    0: 4203,
    1: 4001,
    2: 4005,
}

# model outputs example
output = torch.tensor([0.1, 0.2, 0.7])

TARGET_COLS  = [
    "bowel_healthy","bowel_injury", "extravasation_healthy","extravasation_injury",
    "kidney_healthy", "kidney_low", "kidney_high",
    "liver_healthy", "liver_low", "liver_high",
    "spleen_healthy", "spleen_low", "spleen_high",
]

def get_binary_prediction(output, target_cols):
    # Get the index of the max log-probability
    pred = output.argmax(dim=0, keepdim=True)

    # Convert the label to the original label
    pred = convert_label(pred)

    # Convert the label to binary string
    binary_str = np.binary_repr(pred)
    # 0padding to the length of target columns
    binary_str = binary_str.zfill(len(target_cols))
    print(binary_str)  # Output: 101010

    # Convert binary string to list of digits
    digits = [int(char) for char in binary_str]
    print(digits)  # Output: [1, 0, 1, 0, 1, 0]

    # Create a dictionary of target columns and digits
    data = {col: [digit] for digit, col in zip(digits, target_cols)}

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    return df

