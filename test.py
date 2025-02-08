import matplotlib.pyplot as plt

# Example loss values for training and validation
epochs = range(1, 21)  # 20 epochs
train_loss = [0.9 - (i * 0.03) for i in epochs]  # Simulated training loss
val_loss = [0.95 - (i * 0.025) for i in epochs]  # Simulated validation loss

# Plot the losses
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
