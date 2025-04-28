import matplotlib.pyplot as plt
import pickle

# Load the training history
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

# Plot training & validation accuracy values
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy', fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
plt.subplot(1,2,2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss', fontsize=14)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Optionally, print the final values
print("\nFinal Training Accuracy:", history['accuracy'][-1])
print("Final Validation Accuracy:", history['val_accuracy'][-1])
print("Final Training Loss:", history['loss'][-1])
print("Final Validation Loss:", history['val_loss'][-1])
