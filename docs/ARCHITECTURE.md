# ChordDetect AI: System Architecture

This document provides a detailed overview of the ChordDetect AI system architecture, including its components, data flow, and technical implementation.

## High-Level Architecture

The system follows a machine learning pipeline with three main phases:

```mermaid
graph TD
    A[Audio Input] --> B[Data Generation/Preparation]
    B --> C[Model Training]
    C --> D[Inference]
    D --> E[MIDI Conversion]
    
    style B fill:#f9d5e5,stroke:#333
    style C fill:#eeeeee,stroke:#333
    style D fill:#d0f0c0,stroke:#333
    style E fill:#d0e0f0,stroke:#333
```

## System Components

The system is composed of four main components, each implemented in its own Python file:

### 1. Data Generation (`generate.py`)

Responsible for creating synthetic chord audio samples:

```mermaid
graph TD
    A[Chord Definitions] --> B[Synthetic Sample Generation]
    B -->|Various Chord Types| C[WAV Files]
    B -->|Metadata| D[SQLite Database]
    
    subgraph Variations
        E[Octave Positions]
        F[Inversions]
        G[Voicing Types]
    end
    
    Variations --> B
```

Features:
- Creates samples for 120 chord types (12 roots × 10 chord types)
- Generates multiple variations for each chord:
  - 5 octave positions (very low to very high)
  - Different inversions (root, first, second)
  - 6 voicing types (drop2, spread, cluster, etc.)
- Uses different synthetic timbres
- Adds realistic variations (noise, transients)
- Stores metadata in `chords.db` SQLite database

### 2. Model Training (`train.py`)

Handles feature extraction and model training:

```mermaid
graph TD
    A[Audio Samples] --> B[MFCC Feature Extraction]
    B --> C[Data Split]
    C --> D[CNN Model Training]
    D --> E[Model Evaluation]
    D --> F[Save Trained Model]
    
    subgraph Training Process
        G[Early Stopping]
        H[Learning Rate Reduction]
        I[Validation Monitoring]
    end
    
    Training Process --> D
```

Features:
- Extracts MFCC (Mel-frequency cepstral coefficients) features
- Splits data into training, validation, and test sets
- Implements a CNN architecture for chord classification
- Includes early stopping and learning rate reduction
- Evaluates model performance with accuracy metrics
- Creates confusion matrix visualization
- Saves the trained model to `chord_classifier_model.h5`
- Creates chord mapping in `chord_mapping.csv`

### 3. Inference Pipeline (`infer.py`)

Processes new audio to detect chord progressions:

```mermaid
graph TD
    A[Audio Input] --> B[Source Separation]
    B --> C[Beat Detection]
    C --> D[Audio Segmentation]
    D --> E[MFCC Feature Extraction]
    E --> F[Model Prediction]
    F --> G[Post-processing]
    G --> H[JSON Output]
    
    subgraph Beat Detection Strategies
        I[Enhanced Beat Tracking]
        J[Multi-feature Onset Detection]
        K[Spectral Flux Analysis]
        L[Adaptive Segmentation]
        M[Structural Segmentation]
    end
    
    Beat Detection Strategies --> C
```

Features:
- Uses Demucs for harmonic source separation (if available)
- Implements multiple beat detection strategies with fallbacks
- Segments audio based on detected beats
- Extracts MFCC features matching the training process
- Applies the trained model to classify chord in each segment
- Post-processes results to smooth chord transitions
- Outputs a JSON file with chord progression and timing

### 4. MIDI Conversion (`midi.py`)

Converts JSON chord progression to MIDI:

```mermaid
graph TD
    A[JSON Input] --> B[Chord Name Parsing]
    B --> C[MIDI Note Mapping]
    C --> D[Create MIDI Track]
    D --> E[Add Tempo & Time Signature]
    E --> F[Add Note Events]
    F --> G[Save MIDI File]
```

Features:
- Parses chord names to identify root and type
- Maps chord types to MIDI note combinations
- Creates properly timed MIDI events
- Handles advanced chord types (9th, 11th, suspended, etc.)
- Maintains timing from the original detection
- Outputs a playable MIDI file

## Data Flow

The complete data flow through the system:

```mermaid
flowchart TD
    A[Audio Input] --> B[Source Separation]
    B --> C[Beat Detection]
    C --> D[Audio Segmentation]
    
    subgraph "Feature Extraction"
        D --> E[MFCC Extraction]
        E --> F[Feature Normalization]
    end
    
    subgraph "Chord Classification"
        F --> G[Model Prediction]
        G --> H[Post-processing]
    end
    
    H --> I[JSON Output]
    I --> J[MIDI Conversion]
    J --> K[MIDI Output]
    
    L[Synthetic Chord Generation] -.-> M[Model Training] -.-> G
```

## Technical Details

### Model Architecture

The CNN model architecture used for chord classification:

```mermaid
graph TD
    A[Input Layer: 40×259×1] --> B[Conv2D: 32 filters, 3×3, ReLU]
    B --> C[MaxPooling2D: 2×2]
    C --> D[Dropout: 0.25]
    D --> E[Conv2D: 64 filters, 3×3, ReLU]
    E --> F[MaxPooling2D: 2×2]
    F --> G[Dropout: 0.25]
    G --> H[Conv2D: 128 filters, 3×3, ReLU]
    H --> I[MaxPooling2D: 2×2]
    I --> J[Dropout: 0.25]
    J --> K[Flatten]
    K --> L[Dense: 256, ReLU]
    L --> M[Dropout: 0.5]
    M --> N[Dense: 120, Softmax]
```

### Feature Extraction

MFCC (Mel-frequency cepstral coefficients) extraction process:

```mermaid
graph TD
    A[Audio Segment] --> B[FFT with Window Size 2048]
    B --> C[Convert to Mel Scale]
    C --> D[Apply DCT]
    D --> E[Keep 40 Coefficients]
    E --> F[Normalize Features]
```

### Beat Detection

Multi-strategy approach with fallbacks:

```mermaid
graph TD
    A[Audio Input] --> B[Try Enhanced Beat Tracking]
    B -->|Success| C[Return Beat Locations]
    B -->|Failure| D[Try Multi-feature Onset Detection]
    D -->|Success| C
    D -->|Failure| E[Try Beat Synchronization]
    E -->|Success| C
    E -->|Failure| F[Try Spectral Flux Segmentation]
    F -->|Success| C
    F -->|Failure| G[Try Adaptive Segmentation]
    G -->|Success| C
    G -->|Failure| H[Try Structural Segmentation]
    H -->|Success| C
    H -->|Failure| I[Create Bar-level Segments]
    I --> C
```

## Chord Recognition Process

The recognition process combines multiple techniques for optimal results:

```mermaid
graph TD
    A[Audio Input] --> B[Separate Harmonic Content]
    B --> C[Detect Beats & Segments]
    C --> D[Extract MFCC Features]
    D --> E[Apply Trained CNN Model]
    E --> F[Temporal Smoothing]
    F --> G[Filter Unlikely Transitions]
    G --> H[Final Chord Sequence]
```

## File Structure

```
chord-detect-AI/
├── chord_classifier_model.h5    # Trained CNN model
├── chord_mapping.csv           # Mapping from model output to chord names
├── chord_segments/             # Directory of generated chord samples
├── chords.db                   # SQLite database of chord metadata
├── confusion_matrix.png        # Model evaluation visualization
├── generate.py                 # Script for synthetic sample generation
├── infer.py                    # Chord inference script
├── midi.py                     # MIDI conversion script
├── requirements.txt            # Python dependencies
├── results.json                # Inference results
├── train.py                    # Model training script
└── training_history.png        # Training metrics visualization
```