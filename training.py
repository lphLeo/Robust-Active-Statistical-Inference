import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import xgboost as xgb

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Train function
def train_nn(model, dataloader, epochs=100, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, epochs*len(dataloader))
    loss_fn = torch.nn.SmoothL1Loss()
    
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

def train_tree(X, Y, eta=0.001, max_depth=3, objective='reg:squarederror', boost_rounds=50):
    dtrain = xgb.DMatrix(X, label=Y)
    tree = xgb.train({'eta': eta, 'max_depth': max_depth, 'objective': objective}, dtrain, boost_rounds)
    return tree

def train_tree_2(X, Y):
    xgb_err = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
    tree = xgb_err.fit(X, Y)
    return tree