import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simulated toy cipher parameters
TRUE_KEY = 23  # secret we want to recover
DATA_SIZE = 100

# Fix the random seed for reproducability
torch.manual_seed(0) 
# Generate synthetic plaintexts and their corresponding ciphertexts
plaintexts = torch.randint(0, 256, (DATA_SIZE,), dtype=torch.float32)
ciphertexts = (plaintexts * TRUE_KEY) % 256

# Normalize to [0, 1] for smooth training
plaintexts_norm = plaintexts / 255.0
ciphertexts_norm = ciphertexts / 255.0

# Initialize learnable key parameter
learn_key = torch.tensor([1.0], requires_grad=True)  # start with bad guess

# Optimizer
optimizer = optim.Adam([learn_key], lr=0.1)

# Training loop
loss_history = []
for epoch in range(150):
    optimizer.zero_grad()
    
    # Differentiable cipher simulation: mod 256 is simulated via mod 1 on normalized data
    predictions = (plaintexts_norm * learn_key) % 1.0
    
    # Loss function
    loss = nn.MSELoss()(predictions, ciphertexts_norm)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

# Estimate and rescale key
estimated_key = learn_key.item()
estimated_key_scaled = round(estimated_key * 255)

# Plot loss curve
plt.plot(loss_history)
plt.title("Loss Over Training")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

print(f"🔍 True key: {TRUE_KEY}")
print(f"🎯 Estimated key (scaled): {estimated_key_scaled}")
