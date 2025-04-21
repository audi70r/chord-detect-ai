# Installation Guide

This document provides detailed installation instructions for setting up the ChordDetect AI project.

## Prerequisites

Before installing the project, ensure you have the following:

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ of RAM (8GB+ recommended)
- 2GB+ of free disk space

## Basic Installation

Follow these steps to install the project:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/chord-detect-AI.git
cd chord-detect-AI
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using Python's built-in venv
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

The required dependencies include:

```
numpy>=1.20.0
pydub>=0.25.1
librosa>=0.9.1
matplotlib>=3.5.1
tensorflow>=2.8.0
scikit-learn>=1.0.2
pandas>=1.4.1
pychord>=1.2.2
mido>=1.2.10
python-rtmidi>=1.4.9  # Optional backend for mido
demucs>=4.0.0  # For music source separation
torch>=1.9.0  # Required for demucs
soundfile>=0.10.0  # For audio file handling
```

## Advanced Installation Options

### GPU Acceleration (Recommended for Training)

For faster model training, install TensorFlow with GPU support:

```bash
pip install tensorflow-gpu
```

Requirements:
- NVIDIA GPU with CUDA support
- Appropriate NVIDIA drivers
- CUDA Toolkit 11.2+
- cuDNN 8.1+

Follow the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu) for detailed instructions.

### Source Separation with Demucs

For optimal chord detection, install Demucs for source separation:

```bash
# Install Demucs with PyTorch
pip install demucs
```

On systems with GPUs, this will significantly improve performance.

### MIDI Output Support

For MIDI playback support, install additional packages:

```bash
# For Linux
sudo apt-get install libasound2-dev
pip install python-rtmidi

# For macOS
pip install python-rtmidi

# For Windows
pip install python-rtmidi
```

## Troubleshooting Installation Issues

### LibROSA Installation Issues

If you encounter problems installing LibROSA:

```bash
# On Ubuntu/Debian, install development libraries first
sudo apt-get install -y python3-dev libsndfile1 ffmpeg

# On macOS
brew install libsndfile ffmpeg

# Then install librosa
pip install librosa
```

### CUDA Installation Issues

If you're having issues with GPU support:

```bash
# Verify CUDA installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If it returns an empty list, your GPU is not being recognized. Check:
1. CUDA Toolkit installation
2. cuDNN installation
3. NVIDIA driver compatibility

### Audio Backend Issues

If you encounter audio backend problems:

```bash
# Install system audio libraries
# For Ubuntu/Debian
sudo apt-get install -y libasound2-dev portaudio19-dev

# For macOS
brew install portaudio
```

## Validating Installation

After installation, verify that everything is working:

```bash
# Check if TensorFlow works and recognizes GPU (if available)
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', bool(tf.config.list_physical_devices('GPU')))"

# Check if librosa works
python -c "import librosa; print('LibROSA version:', librosa.__version__)"

# Check if demucs works
python -c "import demucs; print('Demucs available')"
```

## Creating Training Data

After installation, you can generate the training data:

```bash
python generate.py
```

This will create audio samples in the `chord_segments` directory. This process may take some time (10-30 minutes depending on your system).

## Training the Model

To train the model using the generated data:

```bash
python train.py
```

Training typically takes 1-4 hours depending on your hardware and whether GPU acceleration is available.

## Next Steps

After successful installation, refer to the main [README.md](../README.md) for usage instructions.

For a detailed explanation of the system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).