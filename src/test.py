import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)

# Creating the dataset class
class Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.zeros(40, 2)
        self.x[:, 0] = torch.arange(-2, 2, 0.1)
        self.x[:, 1] = torch.arange(-2, 2, 0.1)
        w = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
        b = 1
        func = torch.mm(self.x, w) + b    
        self.y = func + 0.2 * torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]
    # Getter
    def __getitem__(self, idx):          
        return self.x[idx], self.y[idx] 
    # getting data length
    def __len__(self):
        return self.len

# Creating dataset object
data_set = Data()

# Creating a custom Multiple Linear Regression Model
class MultipleLinearRegression(torch.nn.Module):
    # Constructor
    def __init__(self, input_dim, output_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    # Prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Creating the model object
MLR_model = MultipleLinearRegression(2,2)
print("The parameters: ", list(MLR_model.parameters()))

# defining the model optimizer
optimizer = torch.optim.SGD(MLR_model.parameters(), lr=0.22)
# defining the loss criterion
criterion = torch.nn.MSELoss()

# Creating the dataloader
train_loader = DataLoader(dataset=data_set, batch_size=2)

# Train the model
losses = []
epochs = 20
for epoch in range(epochs):
    for x,y in train_loader:
        y_pred = MLR_model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
    print(f"epoch = {epoch}, loss = {loss}")
print("Done training!")

# Plot the losses
plt.plot(losses)
plt.xlabel("no. of iterations")
plt.ylabel("total loss")
plt.show()