from datasets import load_dataset

import torch

n_channels = 4
seq_len = 440

channels = [
    "TP9",
    "FP1",
    "FP2",
    "TP10"
]

dataset = load_dataset("DavidVivancos/MindBigData2022_MNIST_MU")

def reformat(row):
    eeg = torch.FloatTensor(seq_len, n_channels)

    for i, channel in enumerate(channels):
        # get samples for this channel
        eeg[:, i] = torch.FloatTensor([row[f"{channel}-{j}"] for j in range(seq_len)])

    row["eeg"] = eeg
    return row

set_name = "train"
old_columns = dataset[set_name].column_names
old_columns.pop(0)

if __name__ == "__main__":
    dataset = dataset[set_name].map(reformat, num_proc=32)
    dataset.save_to_disk("eeg_mnist_" + set_name)
