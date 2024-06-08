import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to images and cleaned CSV file
images_path = 'Clothing_Dataset_small/images'  # Change to your actual path
cleaned_metadata_path = 'cleaned_metadata.csv'  # Change to your actual path

# Load cleaned CSV file
metadata = pd.read_csv(cleaned_metadata_path)

# Ensure 'id' and 'masterCategory' columns are treated as strings
metadata['id'] = metadata['id'].astype(str) + ".jpg"  # Add the file extension
metadata['masterCategory'] = metadata['masterCategory'].astype(str)

# Ensure all files exist and remove missing files
metadata['file_exists'] = metadata['id'].apply(lambda x: os.path.exists(os.path.join(images_path, x)))
metadata = metadata[metadata['file_exists']]

# Remove classes with fewer than 100 examples
value_counts = metadata['masterCategory'].value_counts()
to_remove = value_counts[value_counts < 100].index
metadata = metadata[~metadata['masterCategory'].isin(to_remove)]

# Split data into training and test sets
train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['masterCategory'])

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['masterCategory']), y=train_df['masterCategory'])
class_weights = dict(zip(np.unique(train_df['masterCategory']), class_weights))

# ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Training data generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=images_path,
    x_col='id',
    y_col='masterCategory',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=images_path,
    x_col='id',
    y_col='masterCategory',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Test data generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=images_path,
    x_col='id',
    y_col='masterCategory',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Transfer learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(metadata['masterCategory'].unique()), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    epochs=15,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size)
print('\nTest accuracy:', test_acc)

# Save the model
model_filename = 'training_custom_dataset_vgg16.keras'
model.save(model_filename)

# Plotting the training history
def plot_training_history(history, model_filename):
    plt.figure(figsize=(12, 4))
    
    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_filename = model_filename.replace('.keras', '_accuracy.png')
    plt.savefig(accuracy_plot_filename)
    
    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_filename = model_filename.replace('.keras', '_loss.png')
    plt.savefig(loss_plot_filename)
    
    plt.show()

# Example usage
plot_training_history(history, model_filename)

# Get true labels and predictions from the test generator
test_generator.reset()
y_true = test_generator.classes

y_pred_proba = model.predict(test_generator, steps=test_generator.n // test_generator.batch_size + 1)
y_pred = np.argmax(y_pred_proba, axis=1)

# Check lengths to debug
print(f'Length of y_true: {len(y_true)}, Length of y_pred: {len(y_pred)}')

# Get the class labels from the generator
class_labels = list(test_generator.class_indices.keys())

# Convert numeric predictions to class labels
y_pred_labels = [class_labels[i] for i in y_pred]

# Classification report
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix created with shape:", cm.shape)
print("Confusion matrix:\n", cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
heatmap_filename = model_filename.replace('.keras', '_confusion_matrix.png')
plt.savefig(heatmap_filename)

# Confirm the heatmap file was saved
if os.path.exists(heatmap_filename):
    print(f"Heatmap saved successfully as {heatmap_filename}")
else:
    print("Error: Heatmap was not saved correctly.")

plt.show()
