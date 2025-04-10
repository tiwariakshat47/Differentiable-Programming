import torch
import matplotlib.pyplot as plt

# --- Setup: Define Cipher Parameters and Generate Data ---
true_key = 23         # The secret key we want to recover
DATA_SIZE = 100       # Number of plaintext-ciphertext pairs

# Fix the random seed for reproducibility
torch.manual_seed(0)

# Generate random plaintexts (values between 0 and 255)
plaintexts = torch.randint(0, 256, (DATA_SIZE,), dtype=torch.float32)

# Apply our toy encryption: ciphertext = (plaintext * true_key) mod 256
ciphertexts = (plaintexts * true_key) % 256

# Normalize values to the range [0, 1] using 256 for consistency with mod256
plaintexts_norm = plaintexts / 256.0
ciphertexts_norm = ciphertexts / 256.0

# --- Initialization: Learnable Key Parameter ---
# Start with an arbitrary guess, here 1.0, and enable gradient tracking.
key = torch.tensor([1.0], requires_grad=True)

# Set learning rate and number of epochs
lr = 0.1
num_epochs = 150

# To record the loss history over training
loss_history = []

# --- Training Loop: Manual Gradient Updates with Autograd ---
for epoch in range(num_epochs):
    # Simulate encryption using the current estimated key.
    # The modulo operation is adjusted to modulo 1.0 for normalized data.
    predictions = (plaintexts_norm * key) % 1.0
    
    # Compute Mean Squared Error (MSE) loss between predictions and true ciphertexts.
    loss = ((predictions - ciphertexts_norm) ** 2).mean()
    
    # Compute gradient of loss with respect to the key manually using autograd.grad
    grad = torch.autograd.grad(loss, key)[0]
    
    # Update the key manually: key = key - lr * grad
    with torch.no_grad():
        key -= lr * grad
    
    # Record loss for visualization
    loss_history.append(loss.item())

# --- Post-Training: Rescale and Report the Key ---
# Since training was done on normalized values (dividing by 256),
# multiply by 256 to get the original scale.
estimated_key = key.item()
estimated_key_scaled = round(estimated_key * 256)

# Plot the loss evolution across epochs
plt.plot(loss_history)
plt.title("Loss Over Training")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

print("🔍 True key:", true_key)
print("🎯 Estimated key (scaled):", estimated_key_scaled)
print("🎯 Estimated key (raw):", estimated_key)