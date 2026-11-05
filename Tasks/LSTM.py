import numpy as np
import tensorflow as tf

# --------------------------------
# 1. Training sentences
# --------------------------------

sentences = [
    "machine learning is useful",
    "machine learning is interesting",
    "machine learning is powerful",
    "deep learning is useful",
    "deep learning is interesting",
    "artificial intelligence is useful",
    "natural language processing is useful",
    "machine learning is important",
    "machine learning is fun"
]

# Combine all sentences
text = " ".join(sentences).lower()

# --------------------------------
# 2. Create character dictionary
# --------------------------------

chars = sorted(set(text))

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

print("Characters:", chars)

# --------------------------------
# 3. Create training sequences
# --------------------------------

seq_length = 4

X = []
y = []

for i in range(len(text) - seq_length):

    # Take 4 characters
    sequence = text[i:i + seq_length]

    # Next character
    next_char = text[i + seq_length]

    # Convert characters to numbers
    X.append([char_to_int[c] for c in sequence])
    y.append(char_to_int[next_char])

X = np.array(X)
y = np.array(y)

# One-hot encode input
X = tf.keras.utils.to_categorical(
    X,
    num_classes=len(chars)
)

print("Training sequences:", len(X))

# --------------------------------
# 4. Create LSTM model
# --------------------------------

model = tf.keras.Sequential([
    
    tf.keras.Input(
        shape=(seq_length, len(chars))
    ),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(
        len(chars),
        activation="softmax"
    )
])

# --------------------------------
# 5. Compile model
# --------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------
# 6. Train model
# --------------------------------

model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

print("\nLSTM model trained successfully!")

# --------------------------------
# 7. Predict next character
# --------------------------------

def predict_next(sequence):

    sequence = sequence.lower()

    # Check length
    if len(sequence) != 4:
        print("Please enter exactly 4 characters.")
        return

    # Check characters
    for c in sequence:
        if c not in char_to_int:
            print("Character not found in training data.")
            return

    # Convert characters to numbers
    input_seq = [
        char_to_int[c] for c in sequence
    ]

    # Convert to array
    input_seq = np.array([input_seq])

    # One-hot encoding
    input_seq = tf.keras.utils.to_categorical(
        input_seq,
        num_classes=len(chars)
    )

    # Prediction
    prediction = model.predict(
        input_seq,
        verbose=0
    )

    # Get character with highest probability
    predicted_char = int_to_char[
        np.argmax(prediction)
    ]

    print("\nInput sequence:", sequence)
    print("Predicted next character:", predicted_char)


# --------------------------------
# 8. User input
# --------------------------------

sequence = input(
    "\nEnter 4 characters: "
)

predict_next(sequence)
