## Standard libraries
import os
import json
import math
import numpy as np
from sklearn.preprocessing import StandardScaler

## Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"

# https://learnopencv.com/multi-label-image-classification-with-pytorch/
# https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440/2
# https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.ipynb


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True  #  bool that, if True, causes cuDNN to only use deterministic convolution algorithms.
torch.backends.cudnn.benchmark = False  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# Fetching the device that will be used throughout
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

"""
Some notes:
epoch = 1, 1 forward and backward cycle on FULL training set
number of iters = number of passes for e/epoch, each pass using [batch size] num samples
batch size = number of training instances chosen in one forward & backward pass
100 samples, BS of 20 --> 100/20 = 5 iters/epoch;
if had 90/20 with drop_last as True in dataloader would get 4 iter/ep for consistent batch size
"""


class car_data(data.Dataset):
    def __init__(self):
        super().__init__()
        self.generate_data()

    def generate_data(self):
        X_train = np.loadtxt(
            "./data/py_training_X.csv", delimiter=",", dtype=np.float32
        )  # float54 used in nb but 32 is what pytorch model takes
        X_test = np.loadtxt("./data/py_training_X.csv", delimiter=",", dtype=np.float32)
        X_features = np.loadtxt(
            "./data/py_X_ft_names.csv", delimiter=",", dtype=np.str_
        )
        y_raw = np.loadtxt("./data/py_y_train.csv", delimiter=",", dtype=np.str_)
        y_enc = np.loadtxt("./data/py_y_enc_train.csv", delimiter=",", dtype=np.float32)
        y_enc_features = np.loadtxt(
            "./data/py_y_enc_ft_names.csv", delimiter=",", dtype=np.str_
        )
        self.listingID = np.loadtxt(
            "./data/py_test_id.csv", delimiter=",", dtype="i8"
        )  # 64-bit signed integer used in nb

        self.X_test = torch.from_numpy(X_test)
        self.X_train = torch.from_numpy(X_train)
        self.n_samples = X_train.shape[0]
        self.y_enc_names = y_enc_features
        self.X_features = X_features

        # for the price
        # need to get the standard location-scale to keep
        # in the same distribution space as sklearn ft we imported
        arr_norm = StandardScaler().fit_transform(
            y_enc[:, -1].reshape(self.n_samples, 1)
        )

        self.price_labels = torch.from_numpy(arr_norm)
        self.trim_labels = torch.from_numpy(
            y_enc[:, :-1]
        )  # trim labels are in the first 29 columns

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {
            "features": self.X_train[idx],
            "labels": {
                "trim_label": self.trim_labels[idx],
                "price_label": self.price_labels[idx],
            },
        }
        return sample

    def get_test_id(self):
        return self.listingID

    def get_feature_names(self):
        return self.X_features, self.y_enc_names

    def get_test_set(self):
        return self.X_test


dataset = car_data()
n_samples = len(dataset)

print("Size of dataset:", n_samples)
print("Data point 0:", dataset[0])

train_size = int(0.8 * n_samples)
test_size = n_samples - train_size
batch_size = 64
learning_rate = 0.0009
n_iterations = math.ceil(n_samples / batch_size)
train_data_loader = data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True
)

# next(iter(...)) catches the first batch of the data loader
# If shuffle is True, this will return a different batch every time we run this cell
# For iterating over the whole dataset, we can simple use "for batch in train_data_loader: ..."
sample = next(iter(train_data_loader))
n_input_features = sample["features"].shape[1]
num_trim_classes = sample["labels"]["trim_label"].shape[1]
print(f"Keys in our sample batch: {sample.keys()}")
print(f"Size for the features in our sample batch: {sample['features'].shape}")
print(
    f"Size for the trim target in our sample batch: {sample['labels']['trim_label'].shape}"
)
print(f"Targets for each trim batch in our sample: {sample['labels']['trim_label']}")
print(
    f"Size for the price target in our sample batch: {sample['labels']['price_label'].shape}"
)
print(f"Targets for each price batch in our sample: {sample['labels']['price_label']}")


# wx+b with sigmoid at the end
class TwoLabelLogRegression(nn.Module):
    def __init__(self, n_input_features, num_trim_classes):
        super(TwoLabelLogRegression, self).__init__()
        self.trim_linear = nn.Linear(n_input_features, num_trim_classes)
        self.price_linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        trim_logits = self.trim_linear(x)
        price_vals = self.price_linear(x)
        # nn.BCEWithLogitsLoss() combines a sigmoid layer and the BCE
        # loss in a single process adjusting for positive numbers
        # applied to the log leading to overflow; this is why we won't
        # apply sigmoid to output of model in this case
        # https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.ipynb
        # note: will note be needed for trim as they are for multiclass and already have 0/1 inputs
        return trim_logits, price_vals


model = TwoLabelLogRegression(n_input_features, num_trim_classes=num_trim_classes)
model.to(device)
print("The parameters: ", list(model.parameters()))

loss_trim = nn.BCEWithLogitsLoss()
loss_price = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_model(model, optimizer, train_data_loader, loss_price, loss_trim, num_epochs=100):
    # Set model to train mode
    model.train()
    # Training loop
    for epoch in range(num_epochs):
        with tqdm(
            total=len(train_data_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for batch_idx, batch in enumerate(train_data_loader):
                ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                features = batch["features"].to(device)
                trim_labels = batch["labels"]["trim_label"].to(device)
                price_labels = batch["labels"]["price_label"].to(device)

                ## Step 2: Run the model on the input data; BCEWithLogitsLoss will apply the
                ## sigmoid for us
                pred_trim_logits, pred_price_vals = model(features)

                ## Step 3: Calculate the loss --> recall -log(p) for BCE means anything near .693
                # is about random aka about 50% samples correctly classified -->
                trim_loss = loss_trim(pred_trim_logits, trim_labels)
                price_loss = loss_price(pred_price_vals, price_labels)

                total_loss = trim_loss + price_loss

                ## Step 4: Perform backpropagation
                # Before calculating the gradients, we need to ensure that they are all zero.
                # The gradients would not be overwritten, but actually added to the existing ones.
                optimizer.zero_grad()
                # Perform backpropagation
                total_loss.backward()

                ## Step 5: Update the parameters
                optimizer.step()


                pbar.update(1)  
                pbar.set_postfix({"Total Loss": total_loss.item(), "Trim Loss": trim_loss.item(), "Price Loss": price_loss.item()})

train_model(model, optimizer, train_data_loader, loss_price, loss_trim)

"""



# Optionally, save the trained model
torch.save(model.state_dict(), "trained_model.pth")

state_dict = model.state_dict()
print(state_dict)

test_dataset = dataset.get_test_set()
# drop_last -> Don't drop the last batch although it is smaller than 128
test_data_loader = data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, drop_last=False
)


def eval_model(model, data_loader):
    model.eval()  # Set model to eval mode
    true_preds, num_preds = 0.0, 0.0

    with torch.no_grad():  # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:
            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)  # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long()  # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

    eval_model(model, test_data_loader)
"""


if __name__ == "__main__":
    pass
