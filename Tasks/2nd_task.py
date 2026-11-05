import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#Step A : Load Dataset
(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32")
x_test = x_test.reshape(-1, 784).astype("float32")

#Normalize
x_train /= 255.0
x_test /= 255.0

#one-hot-Encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Carve out a validation set from training data (last 10,000 samples)
x_val = x_train[-10000:]
y_val = y_train_cat[-10000:]
x_train_final = x_train[:-10000]
y_train_final = y_train_cat[:-10000]

print(f"Train samples: {x_train_final.shape[0]}")
print(f"Validation samples: {x_val.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

#Step B : Fn To build model
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),   # single hidden layer, ReLU
        layers.Dense(10, activation="softmax")  # output layer, softmax over 10 classes
    ])
    return model

#step C: Hyperparameters
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.01   # kept identical across optimizers, as required by the assignment
 
# NOTE: In real-world practice, Adam/RMSprop/Adagrad are usually tuned with
# smaller learning rates (e.g. 0.001) than plain SGD. Forcing the SAME lr
# on all of them (as this assignment requires) is intentionally not
# "each optimizer's best case" -- it isolates the effect of the update
# rule itself, holding everything else constant.

optimizers_to_test = {
    "SGD+Momentum": optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
    "Adagrad": optimizers.Adagrad(learning_rate=LEARNING_RATE),
    "RMSprop": optimizers.RMSprop(learning_rate=LEARNING_RATE),
    "Adam": optimizers.Adam(learning_rate=LEARNING_RATE),
}

#Step D : TRAIN ONE MODEL PER OPTIMIZER, STORE HISTORIES
histories = {}
for name, opt in optimizers_to_test.items():
    print(f"\n{'='*60}\nTraining with optimizer: {name}\n{'='*60}")
 
    model = build_model()
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
 
    history = model.fit(
        x_train_final, y_train_final,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
 
    histories[name] = history.history
 
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"{name} -> Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Step E : plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
# --- Training accuracy ---
for name, h in histories.items():
    axes[0, 0].plot(h["accuracy"], label=name)
axes[0, 0].set_title("Training Accuracy vs Epoch")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True)
 
# --- Validation accuracy ---
for name, h in histories.items():
    axes[0, 1].plot(h["val_accuracy"], label=name)
axes[0, 1].set_title("Validation Accuracy vs Epoch")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)
 
# --- Training loss ---
for name, h in histories.items():
    axes[1, 0].plot(h["loss"], label=name)
axes[1, 0].set_title("Training Loss vs Epoch")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()
axes[1, 0].grid(True)
 
# --- Validation loss ---
for name, h in histories.items():
    axes[1, 1].plot(h["val_loss"], label=name)
axes[1, 1].set_title("Validation Loss vs Epoch")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend()
axes[1, 1].grid(True)
 
plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=150)
plt.show()

#Step F
print(f"\n{'Optimizer':<15}{'Final Train Acc':<18}{'Final Val Acc':<16}{'Final Train Loss':<18}{'Final Val Loss'}")
for name, h in histories.items():
    print(f"{name:<15}{h['accuracy'][-1]:<18.4f}{h['val_accuracy'][-1]:<16.4f}{h['loss'][-1]:<18.4f}{h['val_loss'][-1]:.4f}")
