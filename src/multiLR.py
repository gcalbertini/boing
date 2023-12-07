## Standard libraries
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

## Imports for plotting
import seaborn as sns
from datetime import datetime
from tensorboardX import SummaryWriter, writer
import joblib

sns.set()

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
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
        y_enc = np.loadtxt(
            DATASET_PATH + "/py_y_enc_train.csv", delimiter=",", dtype=np.float32
        )
        y_enc_features = np.loadtxt(
            "./src/transformed_feature_names_y.csv", delimiter=",", dtype=np.str_
        )
        self.listingID = np.loadtxt(
            DATASET_PATH + "/py_test_id.csv", delimiter=",", dtype="i8"
        )  # 64-bit signed integer used in nb

        self.X_test = torch.from_numpy(X_test)
        self.X_train = torch.from_numpy(X_train)
        self.label_names = y_enc_features
        self.n_samples = X_train.shape[0]
        self.X_features = X_features

        # for the price
        # need to get the standard location-scale to keep
        # in the same distribution space as sklearn ft we imported
        scaler = StandardScaler()
        arr_norm = scaler.fit_transform(y_enc[:, -1].reshape(self.n_samples, 1))
        self.mean_value = scaler.mean_
        self.std_dev_value = scaler.scale_

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

    def get_test_set(self):
        return self.X_test

    def get_scaler_params(self):
        return self.mean_value, self.std_dev_value

    def get_trim_label_names(self):
        return self.label_names


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


def train_model(
    model,
    train_data_loader,
    val_data_loader,
    trim_criterion,
    price_criterion,
    optimizer,
    num_epochs=100,
    patience=30,
    max_grad_norm=1,
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

                ## Step 4.5: Gradient clip to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
                writer.add_scalar("Total Loss", total_loss.item(), epoch + 1)
                writer.add_scalar("Trim Loss (CrossEnt)", trim_loss.item(), epoch + 1)
                writer.add_scalar("Price Loss (Huber)", price_loss.item(), epoch + 1)

        # Evaluate on the validation set if provided
        if val_data_loader is not None:
            val_metrics = evaluate_model(model, val_data_loader, price_loss, trim_loss)
            val_loss = val_metrics[
                "Total Loss"
            ]  # Adjust this based on metric for early stopping
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss}\n")
            writer.add_scalar("Validation Loss Epoch", val_loss, epoch + 1)
            writer.add_scalars(
                f"Val_Train_Loss",
                {"Val_Loss_Epoch": val_loss, "Train_Loss": total_loss},
            )

            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= patience or epoch == num_epochs - 1:
                print(f"Saving model weights after epoch {epoch + 1}.")
                # Save the model weights
                model_path = "model_{}_{}".format(timestamp, epoch + 1)
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


def generate_predictions_with_labels(
    model_name, test_set, price_scaler_mean, price_scaler_std, y_enc_features
):
    # Load the pre-trained model weights only if the file exists
    model_weights_path = CHECKPOINT_PATH + model_name
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path))
        model.to(device)
        model.eval()
    else:
        raise NotImplemented

    # Set the model to evaluation mode
    model.eval()

    features = test_set.to(device)
    feature_names = y_enc_features[1:].tolist()

    with torch.no_grad():
        # Forward pass
        pred_trim_logits, pred_price_vals = model(features)

        label_trim = torch.argmax(pred_trim_logits, dim=1).cpu().numpy()
        pred_price_vals = pred_price_vals.cpu().numpy()

        # Reverse the standard scaling for the price labels
        label_price = np.round(
            (pred_price_vals * price_scaler_std) + price_scaler_mean, 2
        )

        # Map the argmax indices to corresponding feature names
        argmax_feature_names = {
            idx: feature_name for idx, feature_name in enumerate(feature_names)
        }

        # Get the string part after "cat__Vehicle_Trim_" for each feature name
        label_trim_feature_names = [
            argmax_feature_names[idx].split("cat__Vehicle_Trim_")[1]
            for idx in label_trim
        ]

    # Set the model back to train mode
    model.train()

    return {
        "trim": label_trim,
        "trim_feature_names": label_trim_feature_names,
        "price": label_price,
        "trim_probs": pred_trim_logits,
        "price_vals": pred_price_vals,
    }


if __name__ == "__main__":
    dataset = car_data()
    n_samples = len(dataset)

    print("Size of dataset:", n_samples)
    print("Data point 0:", dataset[0])

    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    batch_size = 64
    learning_rate = 0.008
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
    print(
        f"Targets for each trim batch in our sample: {sample['labels']['trim_label']}"
    )
    print(
        f"Size for the price target in our sample batch: {sample['labels']['price_label'].shape}"
    )
    print(
        f"Targets for each price batch in our sample: {sample['labels']['price_label']}"
    )

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
    # https://stats.stackexchange.com/questions/313862/huber-loss-on-top-of-cross-entropy
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    print("The parameters: ", list(model.parameters()))

    """
    writer = SummaryWriter()
    train_model(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        trim_criterion=trim_criterion,
        price_criterion=price_criterion,
        optimizer=optimizer,
        num_epochs=1500,
        patience=30,
        max_grad_norm=1,
    )
    writer.close()
    """

    test_data_loader = data.DataLoader(
        dataset=dataset.get_test_set(),
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle the test set during inference
    )

    mu, sigma = dataset.get_scaler_params()

    all_predictions = generate_predictions_with_labels(
        model_name="model_20231207_084237_1500",
        test_set=dataset.get_test_set(),
        price_scaler_std=sigma,
        price_scaler_mean=mu,
        y_enc_features=dataset.get_trim_label_names(),
    )
