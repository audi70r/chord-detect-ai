# ChordDetect AI Tutorial

This tutorial walks you through the entire process of using ChordDetect AI, from installation to analyzing your own songs.

## Quick Start

Here's how to quickly get started with ChordDetect AI:

```mermaid
flowchart LR
    A[Install] --> B[Analyze a Song]
    B --> C[View Results]
    C --> D[Export to MIDI]
```

1. **Install**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Analyze a song**:
   ```bash
   python infer.py your_song.mp3
   ```

3. **Convert to MIDI**:
   ```bash
   python midi.py results.json
   ```

## Complete Workflow

### 1. Installation

First, ensure you have all the required dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/chord-detect-AI.git
cd chord-detect-AI

# Install dependencies
pip install -r requirements.txt
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### 2. Analyzing Your First Song

Let's analyze a song and identify its chord progression:

```bash
# Use the included example file
python infer.py daylight.mp3
```

The process will display information about:
- Audio duration
- Source separation progress
- Beat detection
- Chord identification
- Post-processing

When complete, it will save results to `results.json`.

### 3. Understanding the Results

The `results.json` file contains:

```json
{
  "bpm": 120.5,
  "chords": [
    {"start": 0, "chord": "C"},
    {"start": 2000, "chord": "G"},
    {"start": 4000, "chord": "Am"},
    {"start": 6000, "chord": "F"}
  ]
}
```

- `bpm`: The detected tempo in beats per minute
- `chords`: Array of detected chords with their start times in milliseconds

### 4. Converting to MIDI

To create a MIDI file from the detected chords:

```bash
python midi.py results.json
```

This will create `results.mid`, which you can:
- Play back to hear the chord progression
- Import into a DAW (Digital Audio Workstation)
- Use as a starting point for composition

### 5. Using Different Options

#### Disabling Source Separation

If source separation is taking too long or causing issues:

```bash
python infer.py your_song.mp3 --no-demucs
```

#### Specifying Output Filename

```bash
python infer.py your_song.mp3 --output=song_chords.json
python midi.py song_chords.json song_chords.mid
```

## Practical Examples

### Example 1: Analyzing a Pop Song

```bash
# Analyze a pop song with bar-based segmentation (default)
python infer.py pop_song.mp3

# Convert to MIDI
python midi.py results.json

# The MIDI file can be imported into a DAW for further production
```

### Example 2: Transcribing a Jazz Recording

Jazz recordings often have complex harmonies and may benefit from beat-based segmentation for greater detail:

```bash
# For jazz recordings, beat-based segmentation can capture faster chord changes
python infer.py jazz_recording.mp3 --beat-based

# Check the detected chords
cat results.json

# Convert to MIDI
python midi.py results.json jazz_chords.mid
```

### Example 3: Analyzing Different Time Signatures

For music in different time signatures:

```bash
# For a song in 3/4 time (waltz)
python infer.py waltz_song.mp3 --beats-per-bar=3

# For a song in 6/8 time
python infer.py compound_meter_song.mp3 --beats-per-bar=6

# Convert to MIDI
python midi.py results.json
```

### Example 4: Learning from a Song

```bash
# Analyze a song you want to learn
python infer.py song_to_learn.mp3

# Try both segmentation methods and compare
python infer.py song_to_learn.mp3 --beat-based --output=beat_based_results.json
python midi.py beat_based_results.json beat_based.mid

# Import the MIDI into notation software to create sheet music
```

## Troubleshooting

### Low Confidence Detections

If your results include many "Unknown" chords:

1. Ensure your audio file has clear harmonic content
2. Try a different section of the song
3. Adjust the input volume if the song is too quiet or too loud

```bash
# Process just a section of the song (1:00 to 2:00)
ffmpeg -i original.mp3 -ss 00:01:00 -to 00:02:00 section.mp3
python infer.py section.mp3
```

### Source Separation Issues

If Demucs is crashing or taking too long:

```bash
# Disable source separation
python infer.py your_song.mp3 --no-demucs
```

### Beat Detection Problems

If the chord timing seems off:

1. Ensure the song has a clear rhythmic structure
2. For rubato or tempo-changing music, try analyzing smaller sections

## Advanced Usage

### Comparing Segmentation Methods

To compare the results of different segmentation approaches:

```bash
# Bar-based segmentation (default)
python infer.py your_song.mp3 --output=bar_results.json
python midi.py bar_results.json bar_chords.mid

# Beat-based segmentation
python infer.py your_song.mp3 --beat-based --output=beat_results.json
python midi.py beat_results.json beat_chords.mid

# Compare the two MIDI files in your DAW
```

### Retraining the Model

If you want to customize the model with your own data:

1. Generate custom chord samples:
   ```bash
   # Edit generate.py to focus on your desired chord types
   python generate.py
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Use your new model:
   ```bash
   python infer.py your_song.mp3
   ```

### Customizing MIDI Output

You can edit `midi.py` to customize the MIDI output:

- Change the instrument (program number)
- Adjust the note velocity (loudness)
- Modify chord voicings

For example, to use a piano sound with stronger velocity:

```python
# In midi.py
DEFAULT_VELOCITY = 90  # Increased from 70
DEFAULT_PROGRAM = 0    # Piano
```

## Performance Tips

1. **For faster performance**:
   - Use GPU acceleration if available
   - Process shorter audio files
   - Use `--no-demucs` option if appropriate

2. **For better accuracy**:
   - Use high-quality audio files
   - Ensure your audio has clear harmonic content
   - Songs with clear chord changes work best

## Next Steps

After mastering the basic workflow, explore:

1. Integrating with your music production workflow
2. Batch processing multiple files
3. Creating a custom front-end interface
4. Extending the model to detect more complex chords

For more information, check the other documentation files in the `docs/` directory.