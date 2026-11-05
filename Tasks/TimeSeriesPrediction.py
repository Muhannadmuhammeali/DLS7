import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------
# 1. Load dataset
# -------------------------------

df = pd.read_csv("/content/daily-minimum-temperatures-in-me.csv")

# Convert temperature column to numeric
temperature = pd.to_numeric(
    df["Daily minimum temperatures"],
    errors="coerce"
).values

# Remove missing values
temperature = temperature[~np.isnan(temperature)]

print("Number of temperature values:", len(temperature))

# -------------------------------
# 2. Normalize the data
# -------------------------------

minimum = temperature.min()
maximum = temperature.max()

data = (temperature - minimum) / (maximum - minimum)

# -------------------------------
# 3. Create input sequences
# -------------------------------

sequence_length = 5

X = []
y = []

for i in range(len(data) - sequence_length):

    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------------
# 4. Build LSTM model
# -------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(sequence_length, 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# -------------------------------
# 5. Train the model
# -------------------------------

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    verbose=1
)

# -------------------------------
# 6. Predict
# -------------------------------

predicted = model.predict(X, verbose=0)

# Convert prediction back to temperature
predicted = predicted.flatten()

predicted = predicted * (maximum - minimum) + minimum

actual = y * (maximum - minimum) + minimum

# -------------------------------
# 7. Plot
# -------------------------------

plt.figure(figsize=(12,5))

plt.plot(actual, label="Actual Temperature")
plt.plot(predicted, label="Predicted Temperature")

plt.title("Actual vs Predicted Temperature")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.legend()

plt.show()
