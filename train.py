import os
import numpy as np
import sqlite3
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Settings
CHORD_SEGMENTS_DIR = "chord_segments"
DB_PATH = "chords.db"
SAMPLE_RATE = 44100  # Match the sample rate from generate.py
N_MFCC = 40  # Number of MFCC features to extract
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for FFT

SEGMENT_DURATION = 3.0  # Duration (in seconds) to analyze from each audio file
MODEL_SAVE_PATH = "chord_classifier_model.h5"

# Create a function to load audio files and extract features
def extract_features(file_path, segment_duration=SEGMENT_DURATION):
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path: Path to the audio file
        segment_duration: Duration in seconds to analyze
        
    Returns:
        mfccs: MFCC features
    """
    try:
        # Load audio file (use just the first segment_duration seconds)
        max_duration = int(segment_duration * SAMPLE_RATE)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=segment_duration)
        
        # Pad if audio is shorter than segment_duration
        if len(y) < max_duration:
            y = np.pad(y, (0, max_duration - len(y)), 'constant')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        return mfccs
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Function to load data from database
def load_data_from_db():
    """
    Load data from SQLite database
    
    Returns:
        List of (file_path, chord_name) tuples
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Fetch all records
        cursor.execute("SELECT file_name, chord_name FROM segments")
        data = cursor.fetchall()
        
        # Close connection
        conn.close()
        
        # Convert to full paths
        data = [(os.path.join(CHORD_SEGMENTS_DIR, file_name), chord_name) for file_name, chord_name in data]
        
        return data
    
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return []

# Function to prepare data for training
def prepare_dataset(data, test_size=0.2, val_size=0.2):
    """
    Prepare dataset for training.
    
    Args:
        data: List of (file_path, chord_name) tuples
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        
    Returns:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Label arrays
        label_encoder: LabelEncoder object
    """
    # Initialize lists to store features and labels
    features = []
    labels = []
    
    print(f"Processing {len(data)} audio files...")
    
    # Process each audio file
    for i, (file_path, chord_name) in enumerate(data):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(data)}")
            
        # Extract features
        mfccs = extract_features(file_path)
        
        if mfccs is not None:
            features.append(mfccs)
            labels.append(chord_name)
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Convert features to numpy array
    X = np.array(features)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=y_train)
    
    print(f"Dataset prepared: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

# Function to build the model
def build_model(input_shape, num_classes):
    """
    Build a CNN model for chord classification.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of chord classes
        
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional layers
        layers.Reshape((*input_shape, 1)),  # Add channel dimension
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """
    Train the model.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        history: Training history
    """
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history

# Function to plot training history
def plot_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        label_encoder: LabelEncoder object
    """
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Get chord names
    chord_names = label_encoder.inverse_transform(np.unique(y_test))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=chord_names))
    
    # Plot confusion matrix (for a subset of chords if there are too many)
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # If there are too many classes, plot a subset
    max_classes_to_plot = 20
    if len(chord_names) > max_classes_to_plot:
        # Get most common classes
        class_counts = np.bincount(y_test)
        top_classes = np.argsort(class_counts)[-max_classes_to_plot:]
        
        # Filter data
        mask = np.isin(y_test, top_classes)
        y_test_subset = y_test[mask]
        y_pred_subset = y_pred_classes[mask]
        
        chord_names_subset = label_encoder.inverse_transform(top_classes)
        cm = confusion_matrix(y_test_subset, y_pred_subset)
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Top 20 Classes)')
        plt.colorbar()
        tick_marks = np.arange(len(chord_names_subset))
        plt.xticks(tick_marks, chord_names_subset, rotation=90)
        plt.yticks(tick_marks, chord_names_subset)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(chord_names))
        plt.xticks(tick_marks, chord_names, rotation=90)
        plt.yticks(tick_marks, chord_names)
    
    plt.tight_layout()
    plt.ylabel('True Chord')
    plt.xlabel('Predicted Chord')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Main function
def main():
    print("=== Chord Classification Model Training ===")
    
    # Check if data directory exists
    if not os.path.exists(CHORD_SEGMENTS_DIR):
        print(f"Error: Directory '{CHORD_SEGMENTS_DIR}' not found. Please run generate.py first.")
        return
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database '{DB_PATH}' not found. Please run generate.py first.")
        return
    
    # Load data from database
    print("Loading data from database...")
    data = load_data_from_db()
    
    if not data:
        print("No data found. Please check database and audio files.")
        return
    
    # Prepare dataset
    print("Preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_dataset(data)
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1:] 
    num_classes = len(label_encoder.classes_)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Chord classes: {label_encoder.classes_}")
    
    # Save class mapping
    class_mapping = pd.DataFrame({
        'index': range(len(label_encoder.classes_)),
        'chord': label_encoder.classes_
    })
    class_mapping.to_csv('chord_mapping.csv', index=False)
    print("Chord mapping saved to chord_mapping.csv")
    
    # Build model
    print("Building model...")
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Train model
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    print("Training complete!")

# Run the script
if __name__ == "__main__":
    main()