# train_gender_model.py
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GenderModelTrainer:
    def __init__(self):
        self.img_width, self.img_height = 128, 128
        self.batch_size = 32
        self.epochs = 50
        self.train_data_dir = 'data/train'
        self.validation_data_dir = 'data/validation'  # Fixed typo from 'viladation'
        self.model_save_path = 'models/custom_gender_model.h5'
        self.class_names = ['male', 'female']
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Verify datasets before training
        if not self.verify_datasets():
            raise ValueError("Dataset verification failed. Cannot proceed with training.")
        
        # Clean datasets before training
        self.clean_datasets()
    
    def verify_datasets(self):
        """Verify that the dataset structure is correct"""
        print("\nVerifying dataset structure...")
        
        valid = True
        
        for dataset in [self.train_data_dir, self.validation_data_dir]:
            if not os.path.exists(dataset):
                print(f"Error: Dataset directory not found: {dataset}")
                valid = False
                continue
                
            for class_name in self.class_names:
                class_dir = os.path.join(dataset, class_name)
                if not os.path.exists(class_dir):
                    print(f"Error: Class directory not found: {class_dir}")
                    valid = False
                    continue
                    
                num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"Found {num_images} images in {class_dir}")
                
                if num_images < 10:  # Minimum number of images per class
                    print(f"Warning: Low number of images ({num_images}) in {class_dir}")
        
        return valid
    
    def clean_datasets(self):
        """Remove corrupt images from training and validation sets"""
        print("\nChecking for corrupt images...")
        for dataset in [self.train_data_dir, self.validation_data_dir]:
            for class_name in self.class_names:
                class_dir = os.path.join(dataset, class_name)
                if not os.path.exists(class_dir):
                    continue
                    
                for filename in tqdm(os.listdir(class_dir), desc=f"Checking {class_name}"):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        # Verify image is not corrupt
                        img = Image.open(filepath)
                        img.verify()
                        img.close()
                        
                        # Convert to RGB if needed
                        img = Image.open(filepath)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(filepath)
                        img.close()
                    except (IOError, SyntaxError, OSError) as e:
                        print(f"Removing corrupt image: {filepath}")
                        os.remove(filepath)
    
    def create_model(self):
        """Create a custom CNN model for gender classification"""
        model = Sequential()
        
        # Convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_width, self.img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='binary',
            classes=self.class_names
        )
        
        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='binary',
            classes=self.class_names
        )
        
        return train_generator, validation_generator
    
    def train_model(self):
        """Train the gender classification model"""
        model = self.create_model()
        train_generator, validation_generator = self.prepare_data()
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            self.model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        print("\nStarting training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        # Save the final model
        model.save(self.model_save_path)
        print(f"\nModel saved to {self.model_save_path}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
    
    def evaluate_model(self, model):
        """Evaluate the trained model"""
        print("\nEvaluating model...")
        _, validation_generator = self.prepare_data()
        results = model.evaluate(validation_generator)
        print(f"Validation Loss: {results[0]:.4f}")
        print(f"Validation Accuracy: {results[1]:.4f}")

if __name__ == "__main__":
    trainer = GenderModelTrainer()
    trained_model = trainer.train_model()
    trainer.evaluate_model(trained_model)