import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Enable eager execution mode for debugging
tf.config.run_functions_eagerly(True)

# Define paths
image_dir = r'Thesis\RoadSegmentation\dataset\images'
mask_dir = r'Thesis\RoadSegmentation\dataset\masks'

# Parameters
img_height = 512
img_width = 512
img_channels = 3
batch_size = 4  # Reduced batch size to handle more complex model
epochs = 100

def load_data(image_dir, mask_dir, img_height, img_width):
    images = []
    masks = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        image = load_img(img_path, target_size=(img_height, img_width), color_mode='rgb')
        mask = load_img(mask_path, target_size=(img_height, img_width), color_mode='rgb')

        image = img_to_array(image)
        mask = img_to_array(mask)

        images.append(image)
        masks.append(mask)

    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0

    return images, masks

# Load data
images, masks = load_data(image_dir, mask_dir, img_height, img_width)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

def unet_model(input_size=(512, 512, 3)):
    inputs = tf.keras.layers.Input(input_size)

    def conv_block(input_tensor, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def encoder_block(input_tensor, num_filters):
        x = conv_block(input_tensor, num_filters)
        p = tf.keras.layers.MaxPooling2D((2, 2))(x)
        p = tf.keras.layers.Dropout(0.3)(p)
        return x, p

    def decoder_block(input_tensor, skip_tensor, num_filters):
        x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        x = tf.keras.layers.concatenate([x, skip_tensor])
        x = tf.keras.layers.Dropout(0.3)(x)
        x = conv_block(x, num_filters)
        return x

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(d4)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Create U-Net model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.summary()

# Data augmentation (optional)
data_gen_args = dict(rotation_range=10.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# Use the same seed for image and mask generators
seed = 1
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

# Create image and mask generators
image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

# Combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Save the model
model.save('roadSegmentationUnet.keras')
