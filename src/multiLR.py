## Standard libraries
import os
import json
import math
import numpy as np
from sklearn.preprocessing import StandardScaler

## Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import softmax
from datetime import datetime

sns.set()

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

DATASET_PATH = "./data/"
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
            DATASET_PATH + "/py_training_X.csv", delimiter=",", dtype=np.float32
        )  # float54 used in nb but 32 is what pytorch model takes
        X_test = np.loadtxt(
            DATASET_PATH + "/py_training_X.csv", delimiter=",", dtype=np.float32
        )
        X_features = np.loadtxt(
            DATASET_PATH + "/py_X_ft_names.csv", delimiter=",", dtype=np.str_
        )
        y_raw = np.loadtxt(
            DATASET_PATH + "/py_y_train.csv", delimiter=",", dtype=np.str_
        )
        y_enc = np.loadtxt(
            DATASET_PATH + "/py_y_enc_train.csv", delimiter=",", dtype=np.float32
        )
        y_enc_features = np.loadtxt(
            DATASET_PATH + "/py_y_enc_ft_names.csv", delimiter=",", dtype=np.str_
        )
        self.listingID = np.loadtxt(
            DATASET_PATH + "/py_test_id.csv", delimiter=",", dtype="i8"
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
val_size = n_samples - train_size
train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

batch_size = 2**7
learning_rate = 0.0001
weight_decay = 0
momentum = 0


train_data_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle the training set during training
)

val_data_loader = data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,  # Do not shuffle the validation set during validation
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
class MLT_log_reg(nn.Module):
    def __init__(self, n_input_features, num_trim_classes):
        super(MLT_log_reg, self).__init__()
        self.trim_linear = nn.Linear(n_input_features, num_trim_classes)
        self.price_linear = nn.Linear(n_input_features, 1)
        # Fight off co-adaptation (when multiple neurons in a layer extract the same, or very similar, hidden features from the input data)
        # So p% of nodes in input and hidden layer dropped in every iteration (batch) --> sparsity
        # By using dropout, in every iteration, you will work on a smaller neural network
        # than the previous one and, therefore, approaches regularization which (if using MSE)
        # shrinks the squared norm of the weights --> reduce overfitting

        # Define learnable parameters for trim_weight and price_weight
        # https://arxiv.org/abs/2004.13379
        self.trim_weight = nn.Parameter(torch.Tensor([1.0]))
        self.price_weight = nn.Parameter(torch.Tensor([1.0]))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(x)
        trim_non_logits = self.trim_linear(x)
        price_vals = self.price_linear(x)
        # cross entropy loss will normalize the logits across class
        return torch.sigmoid(trim_non_logits), price_vals


model = MLT_log_reg(n_input_features, num_trim_classes=num_trim_classes)
model.to(device)

# NOTE reduction by mean default means loss will be normalized by batch size
# to get a loss per epoch can set to sum or mult by batch size and then divide by
# entire dataset size
# CrossEntropyLoss combines nn.LogSoftmax() in model followed by nn.NLLLoss() criterion in one single go assuming input is logits (softmax'd)
# The softmax function returns probabilities between [0, 1] where sum *across classes* sum to 1; log of these probabilities
# returns values between [-inf, 0], since log(0) = -inf and log(1) = 0
trim_criterion = nn.CrossEntropyLoss()  # need input in logits
price_criterion = (
    nn.HuberLoss()
)  # see notes in EDA; we did not sift out outliers in totality for hetereroscedastic priors
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
)
print("The parameters: ", list(model.parameters()))


def train_model(
    model,
    optimizer,
    train_data_loader,
    price_loss,
    trim_loss,
    num_epochs=50,
    val_data_loader=val_data_loader,
    patience=25,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set model to train mode
    model.train()

    # Initialize early stopping parameters
    best_loss = float("inf")
    consecutive_no_improvement = 0

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

                # Access learnable parameters
                trim_weight = model.trim_weight
                price_weight = model.price_weight

                ## Step 3: Calculate the loss --> recall -log(p) for BCE means anything near .693
                # is about random aka about 50% samples correctly classified
                trim_loss = trim_criterion(
                    pred_trim_logits, trim_labels
                )  # loss we chose here will apply softmax to logits then get negative log likelihood loss
                price_loss = price_criterion(pred_price_vals, price_labels)

                total_loss = (
                    price_loss / (2 * price_weight**2)
                    + trim_loss / (2 * trim_weight**2)
                    + torch.log(trim_weight * price_weight)
                )

                ## Step 4: Perform backpropagation
                # Before calculating the gradients, we need to ensure that they are all zero.
                # The gradients would not be overwritten, but actually added to the existing ones.
                optimizer.zero_grad()
                total_loss.backward()

                ## Step 5: Update the parameters
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Total Loss": total_loss.item(),
                        "Trim Loss (CrossEnt)": trim_loss.item(),
                        "Price Loss (Huber)": price_loss.item(),
                    }
                )

        # Evaluate on the validation set if provided
        if val_data_loader is not None:
            val_metrics = evaluate_model(model, val_data_loader, price_loss, trim_loss)
            val_loss = val_metrics[
                "Total Loss"
            ]  # Adjust this based on metric for early stopping
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss}\n")

            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= patience or epoch == num_epochs - 1:
                print(f"Saving model weights after epoch {epoch + 1}.")
                # Save the model weights
                model_path = "model_{}_{}".format(timestamp, epoch)
                if not os.path.exists(CHECKPOINT_PATH):
                    os.makedirs(CHECKPOINT_PATH)
                torch.save(model.state_dict(), CHECKPOINT_PATH + model_path)
                break


def evaluate_model(model, data_loader, price_loss, trim_loss):
    # Set model to evaluation mode
    model.eval()

    total_price_loss = 0.0
    total_trim_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            features = batch["features"].to(device)
            trim_labels = batch["labels"]["trim_label"].to(device)
            price_labels = batch["labels"]["price_label"].to(device)

            # Forward pass
            pred_trim_logits, pred_price_vals = model(features)

            # Calculate losses
            trim_loss = trim_criterion(pred_trim_logits, trim_labels)
            price_loss = price_criterion(pred_price_vals, price_labels)

            # https://stackoverflow.com/questions/54053868/how-do-i-get-a-loss-per-epoch-and-not-per-batch
            total_trim_loss += trim_loss.item()
            total_price_loss += price_loss.item()

    # Set model back to train mode
    model.train()

    # Return average losses over the validation set
    avg_trim_loss = total_trim_loss / len(data_loader)
    avg_price_loss = total_price_loss / len(data_loader)

    return {
        "Trim Loss (CrossEnt)": avg_trim_loss,
        "Price Loss (Huber)": avg_price_loss,
        "Total Loss": avg_trim_loss + avg_price_loss,
    }


# Example usage:
train_model(
    model,
    optimizer,
    train_data_loader,
    price_criterion,
    trim_criterion,
    num_epochs=10,
)

# val_metrics = evaluate_model(model, val_data_loader, price_loss, trim_loss)
# print(f"Validation Metrics: {val_metrics}")


if __name__ == "__main__":
    pass
