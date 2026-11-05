import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# -------------------------------
# 1. Load and preprocess MNIST
# -------------------------------
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.

# Add channel dimension (for Conv2D)
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test,  axis=-1)

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# -------------------------------
# 2. Add Gaussian noise
# -------------------------------
noise_factor = 0.5   # controls noise level
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy  = x_test  + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to stay in [0,1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy,  0., 1.)

# -------------------------------
# 3. Build the Denoising Autoencoder
# -------------------------------
input_img = layers.Input(shape=(28, 28, 1))

# ---------- Encoder ----------
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)          # 14x14x32
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)          # 7x7x64
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)          # 4x4x128 (bottleneck)

# ---------- Decoder ----------
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)                          # 8x8x128
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)                          # 16x16x64
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)                          # 32x32x32 (crop later? we'll resize)

# Final layer to produce 28x28x1
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Crop or resize to 28x28 if needed – but UpSampling 2x three times gives 32x32,
# so we need to crop or adjust. Better: use strides or adjust padding.
# To keep simple, we'll use a slightly different architecture:
# Use Conv2DTranspose with stride=2 to get exact dimensions.

# Let's re‑define the decoder more carefully to get 28x28.
# We'll use Conv2DTranspose with strides.

# Better model:
def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))
    
    # Encoder: two maxpool layers => 7x7 bottleneck
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)   # 14x14
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)   # 7x7
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    # bottleneck: 7x7x128

    # Decoder: transposed conv to go back to 28x28
    x = layers.Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding='same')(x)  # 14x14
    x = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)  # 28x28
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)              # 28x28

    return models.Model(input_img, decoded)

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# -------------------------------
# 4. Train the model
# -------------------------------
history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=20,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# -------------------------------
# 5. Evaluate and visualize
# -------------------------------
decoded_imgs = autoencoder.predict(x_test_noisy)

# Show 10 examples
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # Original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Noisy
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(x_test_noisy[i].squeeze(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')
    
    # Denoised
    ax = plt.subplot(3, n, i+2*n+1)
    plt.imshow(decoded_imgs[i].squeeze(), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Compare reconstruction quality (PSNR)
# -------------------------------
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Compute average PSNR over test set
psnr_noisy = []
psnr_denoised = []
for i in range(len(x_test)):
    psnr_noisy.append(psnr(x_test[i], x_test_noisy[i]))
    psnr_denoised.append(psnr(x_test[i], decoded_imgs[i]))
print(f"Average PSNR (noisy): {np.mean(psnr_noisy):.2f} dB")
print(f"Average PSNR (denoised): {np.mean(psnr_denoised):.2f} dB")
