import torch
import pandas as pd

# Load CSV file
df = pd.read_csv("./C3L/I2C.csv", header=None)  # header=None if no column names

# Convert to PyTorch tensor
tensor = torch.tensor(df.values, dtype=torch.float32)
print(tensor.shape)
torch.save(tensor, "./C3L/I2C.pt")  # Save tensor to file
