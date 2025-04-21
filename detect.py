#!/usr/bin/env python3
"""
detect.py - Detect chord progressions using Fourier transform analysis

This script provides an alternative chord detection approach that relies on
direct frequency analysis instead of machine learning. It analyzes audio files
using Fast Fourier Transform (FFT) to identify chord progressions.

Usage:
    python detect.py path/to/audio_file.mp3 [options]

Options:
    --output=<file_path>      Specify output JSON file path (default: results_fft.json)
    --beat-based              Use beat-based segmentation (default is bar-based)
    --beats-per-bar=<num>     Specify beats per bar (default is 4 for 4/4 time)
    --no-demucs               Disable Demucs source separation
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import librosa
import soundfile as sf
import tempfile
import shutil
import random
from collections import Counter
from scipy import signal
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import demucs
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    print("Warning: Demucs not available. Install with 'pip install demucs'")
    DEMUCS_AVAILABLE = False

# Constants
SAMPLE_RATE = 44100
FFT_SIZE = 8192  # Large FFT size for better frequency resolution
HOP_LENGTH = 512
OUTPUT_PATH = "results_fft.json"

# Demucs settings
DEMUCS_MODEL = "htdemucs"  # The pre-trained model to use
DEMUCS_SEGMENT = 8  # Length of each segment in seconds when processing with demucs
DEMUCS_OVERLAP = 0.25  # Overlap between segments

# Define note frequencies (A4 = 440Hz is MIDI note 69)
# Frequency table from C1 (MIDI 24) to B8 (MIDI 107)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_FREQUENCIES = {}

# Generate frequency table for all notes
def generate_note_frequencies():
    # A4 = 440Hz = MIDI note 69
    for midi_note in range(24, 108):
        note_name = NOTE_NAMES[midi_note % 12]
        octave = midi_note // 12 - 1
        frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        full_note_name = f"{note_name}{octave}"
        NOTE_FREQUENCIES[full_note_name] = frequency

# Generate note frequencies at initialization
generate_note_frequencies()

# Chord templates - define pitch class sets for different chord types
# These are binary representations of the chord notes (1 = note present, 0 = note absent)
CHORD_TEMPLATES = {
    # Major chords
    "major": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C, E, G
    "minor": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C, Eb, G
    "7":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # C, E, G, Bb
    "maj7":  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # C, E, G, B
    "m7":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # C, Eb, G, Bb
    "dim":   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # C, Eb, Gb
    "aug":   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # C, E, G#
    "sus4":  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # C, F, G
    "sus2":  [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # C, D, G
    "9":     [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # C, D, E, G, Bb
    "add9":  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C, D, E, G
}

def detect_beats(audio, sr):
    """
    Detect beats in an audio file using multiple advanced strategies.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        
    Returns:
        beats: Beat positions in samples
        beat_times: Beat positions in seconds
        tempo: Estimated tempo in BPM
    """
    try:
        # Pre-process audio to improve beat detection
        # 1. Normalize audio
        audio = librosa.util.normalize(audio)
        
        # 2. Apply high-pass filter to remove low frequency noise
        audio_filtered = librosa.effects.preemphasis(audio)
        
        # 3. Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(audio_filtered)
        
        print("Analyzing audio characteristics...")
        duration = len(audio) / sr
        print(f"Audio duration: {duration:.2f} seconds")
        
        # First, get a reliable tempo estimate using multiple methods
        print("Estimating tempo...")
        
        # Method 1: Standard tempo estimation on percussive component
        tempo_standard = librosa.beat.tempo(y=y_percussive, sr=sr)[0]
        
        # Method 2: Tempo from autocorrelation
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        tempo_autocorr = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Method 3: Tempogram-based estimation
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempo_tempogram = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                            aggregate=np.median)[0]
        
        # Choose most reliable tempo estimate by comparing methods
        tempos = [tempo_standard, tempo_autocorr, tempo_tempogram]
        tempo_estimate = np.median(tempos)
        
        print(f"Tempo estimates: Standard={tempo_standard:.1f}, Autocorr={tempo_autocorr:.1f}, Tempogram={tempo_tempogram:.1f}")
        print(f"Selected tempo: {tempo_estimate:.1f} BPM")
        
        # Strategy 1: Enhanced beat tracking with better tempo initialization
        print("Trying enhanced beat tracking...")
        beat_frames = librosa.beat.beat_track(
            y=y_percussive,  # Use percussive component for beat tracking
            sr=sr,
            start_bpm=tempo_estimate,
            tightness=400,  # Higher tightness forces tempo to be closer to estimate
            trim=False
        )[1]
        
        if len(beat_frames) > 5:  # If we have a reasonable number of beats
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            beats = librosa.time_to_samples(beat_times, sr=sr)
            print(f"Enhanced beat tracking successful: {len(beats)} beats at {tempo_estimate:.1f} BPM")
            return beats, beat_times, tempo_estimate
        
        # Strategy 2: Multi-feature onset detection
        print("Enhanced beat tracking failed, trying multi-feature onset detection...")
        
        # Compute multiple onset strength signals and combine them
        onset_env_hfc = librosa.onset.onset_strength(
            y=audio, sr=sr, feature=librosa.feature.spectral_flux)
        onset_env_energy = librosa.onset.onset_strength(
            y=audio, sr=sr, feature=librosa.feature.rms)
        onset_env_spectral = librosa.onset.onset_strength(
            y=audio, sr=sr, feature=librosa.feature.spectral_flatness)
        
        # Combine onset detectors (weighted sum)
        combined_onset = (onset_env + onset_env_hfc + onset_env_energy + onset_env_spectral) / 4
        
        # Dynamic thresholding - adapt to audio characteristics
        threshold_range = np.linspace(0.3, 1.5, 10)  # More fine-grained thresholds
        for threshold in threshold_range:
            # Scale threshold by local statistics
            local_avg = np.mean(combined_onset)
            local_std = np.std(combined_onset)
            adaptive_threshold = local_avg + threshold * local_std
            
            # Find peaks in onset strength signal
            onset_peaks = []
            min_distance = int(sr / HOP_LENGTH * 60 / (tempo_estimate * 4))  # Minimum distance between peaks
            
            # Sliding window peak picking with adaptive threshold
            for i in range(1, len(combined_onset) - 1):
                if (combined_onset[i] > adaptive_threshold and
                    combined_onset[i] > combined_onset[i-1] and
                    combined_onset[i] >= combined_onset[i+1]):
                    
                    # Check if it's a sufficiently isolated peak
                    if not onset_peaks or i - onset_peaks[-1] >= min_distance:
                        onset_peaks.append(i)
            
            if len(onset_peaks) > 10:  # If we have enough peaks
                onset_times = librosa.frames_to_time(onset_peaks, sr=sr, hop_length=HOP_LENGTH)
                onset_samples = librosa.time_to_samples(onset_times, sr=sr)
                
                print(f"Multi-feature onset detection successful: {len(onset_peaks)} onsets")
                print(f"Using tempo estimate: {tempo_estimate:.1f} BPM")
                return onset_samples, onset_times, tempo_estimate
        
        # Strategy 3: Beat tracking with beat synchronization
        print("Multi-feature onset detection failed, trying beat synchronization...")
        
        # Try different tightness values for more robust tracking
        tightness_values = [100, 200, 300, 400, 500]
        for tightness in tightness_values:
            # Get beat positions using different tightness values
            tempo, beat_frames = librosa.beat.beat_track(
                y=y_percussive, 
                sr=sr,
                start_bpm=tempo_estimate,
                tightness=tightness
            )
            
            if len(beat_frames) > 5:
                # Use the initially estimated tempo for consistency
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                beats = librosa.time_to_samples(beat_times, sr=sr)
                print(f"Beat tracking with tightness={tightness} successful: {len(beats)} beats")
                return beats, beat_times, tempo_estimate
        
        # Strategy 4: Spectral flux + novelty-based segmentation
        print("Beat synchronization failed, trying spectral flux segmentation...")
        
        # Compute spectral flux
        S = np.abs(librosa.stft(y_percussive))
        onset_sf = librosa.onset.onset_strength(S=S, sr=sr)
        
        # Compute spectral novelty
        onset_novelty = librosa.onset.onset_strength(y=y_percussive, sr=sr, 
                                                  feature=librosa.feature.spectral_novelty)
        
        # Combine flux and novelty
        combined_feature = onset_sf + onset_novelty
        
        # Find significant peaks
        peaks = librosa.util.peak_pick(combined_feature, 
                                      pre_max=3, post_max=3, 
                                      pre_avg=10, post_avg=10, 
                                      delta=0.3, wait=30)
        
        if len(peaks) > 5:
            peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH)
            peak_samples = librosa.time_to_samples(peak_times, sr=sr)
            print(f"Spectral flux segmentation successful: {len(peaks)} segments")
            return peak_samples, peak_times, tempo_estimate
        
        # Strategy 5: Adaptive segmentation using dynamic time warping
        print("Trying adaptive segmentation...")
        
        # Generate a periodic grid at the estimated tempo
        grid_frames = np.arange(0, len(onset_env), sr * 60 / (tempo_estimate * HOP_LENGTH))
        grid_frames = grid_frames.astype(int)[grid_frames < len(onset_env)]
        
        if len(grid_frames) > 0:
            # Use dynamic time warping to align grid to audio features
            aligned_frames = grid_frames
            
            # Adjust each grid point to nearest peak in onset envelope
            search_radius = int(sr * 0.1 / HOP_LENGTH)  # 100ms search radius
            for i in range(len(aligned_frames)):
                idx = aligned_frames[i]
                if idx < len(onset_env):
                    # Find local maximum within search window
                    start = max(0, idx - search_radius)
                    end = min(len(onset_env), idx + search_radius + 1)
                    if start < end:
                        local_idx = np.argmax(onset_env[start:end])
                        aligned_frames[i] = start + local_idx
            
            # Convert to time and samples
            segment_times = librosa.frames_to_time(aligned_frames, sr=sr, hop_length=HOP_LENGTH)
            segment_samples = librosa.time_to_samples(segment_times, sr=sr)
            
            print(f"Adaptive segmentation successful: {len(segment_times)} segments")
            return segment_samples, segment_times, tempo_estimate
        
        # Strategy 6: Structural segmentation
        print("Trying structural segmentation...")
        
        # Compute a log-scaled mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Compute a structural segmentation
        bounds = librosa.segment.agglomerative(log_mel_spec, k=10)  # Up to 10 segments
        bound_times = librosa.frames_to_time(bounds, sr=sr)
        bound_samples = librosa.time_to_samples(bound_times, sr=sr)
        
        if len(bound_times) > 3:  # If we found at least a few structural boundaries
            print(f"Structural segmentation successful: {len(bound_times)} segments")
            return bound_samples, bound_times, tempo_estimate
        
        # If all else fails, create smart segments based on estimated tempo
        print("All beat detection methods failed. Creating musically-informed segments.")
        
        # Use the reliable tempo estimate from earlier
        beats_per_bar = 4  # Assume 4/4 time signature
        # Create segments at bar level (every 4 beats typically)
        seconds_per_beat = 60 / tempo_estimate
        seconds_per_bar = seconds_per_beat * beats_per_bar
        
        # Create segments at bar boundaries
        num_bars = int(duration / seconds_per_bar)
        if num_bars == 0:
            num_bars = 1
        
        segment_times = np.arange(0, duration, seconds_per_bar)[:num_bars+1]
        segment_samples = librosa.time_to_samples(segment_times, sr=sr)
        
        print(f"Created {len(segment_samples)} bar-level segments at {tempo_estimate:.1f} BPM")
        return segment_samples, segment_times, tempo_estimate
        
    except Exception as e:
        print(f"Error in all beat detection methods: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency fallback - should almost never be needed now
        print("CRITICAL: Using emergency fallback with fixed segments")
        duration = len(audio) / sr
        
        # Even in emergency mode, try to get a tempo estimate
        try:
            tempo_guess = librosa.beat.tempo(
                y=librosa.util.normalize(audio), 
                sr=sr,
                aggregate=None
            )[0]
            if 60 <= tempo_guess <= 200:
                emergency_tempo = tempo_guess
            else:
                emergency_tempo = 120
        except:
            emergency_tempo = 120
            
        seconds_per_beat = 60 / emergency_tempo
        # Create at least 16 segments, regardless of duration
        num_segments = max(16, int(duration / seconds_per_beat))
        segment_times = np.linspace(0, duration, num_segments)
        segment_samples = librosa.time_to_samples(segment_times, sr=sr)
        
        print(f"Emergency fallback: {len(segment_samples)} segments at {emergency_tempo:.1f} BPM")
        return segment_samples, segment_times, emergency_tempo

def detect_chroma(audio_segment, sr=SAMPLE_RATE):
    """
    Extract chroma features from an audio segment using FFT.
    The FFT size is dynamically adjusted based on the segment length.
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate
        
    Returns:
        chroma: Chroma feature vector
    """
    # Determine optimal FFT size based on the length of the segment
    # We want to use the entire segment for frequency analysis
    # but rounded to a power of 2 for efficient FFT
    segment_length = len(audio_segment)
    
    # Get the next power of 2 that is greater than segment_length
    # This ensures we capture the full bar in our FFT
    fft_size = 2 ** int(np.ceil(np.log2(segment_length)))
    
    # Cap at a reasonable maximum size
    fft_size = min(fft_size, 2**16)  # Max of 65536 points
    
    # Set a minimum size for very short segments
    fft_size = max(fft_size, 4096)  # Min of 4096 points for frequency resolution
    
    # Only print FFT size occasionally to avoid flooding the console
    if random.random() < 0.05:  # Print for ~5% of segments
        print(f"Using FFT size of {fft_size} for segment of length {segment_length}")
    
    # Apply Hann window to avoid spectral leakage
    windowed_segment = audio_segment * signal.windows.hann(len(audio_segment))
    
    # Calculate FFT (zero padding to fft_size)
    fft_result = np.abs(np.fft.rfft(windowed_segment, n=fft_size))
    
    # Get frequency bins
    freqs = np.fft.rfftfreq(fft_size, 1/sr)
    
    # Initialize 12-bin chroma vector (one bin per semitone)
    chroma = np.zeros(12)
    
    # Map FFT bins to pitch classes (C, C#, D, etc.)
    for i, magnitude in enumerate(fft_result):
        if i < len(freqs) and freqs[i] > 0:  # Avoid DC component
            # Convert frequency to pitch class (0-11)
            pitch_class = int(round(12 * np.log2(freqs[i] / 440.0) + 69)) % 12
            # Add magnitude to the corresponding pitch class
            chroma[pitch_class] += magnitude
    
    # Normalize the chroma vector
    chroma_norm = np.linalg.norm(chroma)
    if chroma_norm > 0:
        chroma = chroma / chroma_norm
    
    return chroma

def identify_chord(chroma, segment_duration=None):
    """
    Identify the chord from a chroma vector using template matching.
    
    Args:
        chroma: Normalized chroma vector
        segment_duration: Optional duration of the segment in seconds,
                          used to adjust confidence threshold
        
    Returns:
        chord_name: Identified chord name
    """
    # Rotate chroma vector to all possible keys and compare with templates
    best_score = -1
    best_chord = "N.C."  # No chord
    
    # Store all candidates for further analysis
    candidates = []
    
    for root in range(12):  # 12 possible roots (C, C#, D, etc.)
        # Rotate chroma to create the root alignment
        rotated_chroma = np.roll(chroma, -root)
        
        for chord_type, template in CHORD_TEMPLATES.items():
            # Calculate correlation between rotated chroma and template
            # Using dot product as similarity measure
            score = np.dot(rotated_chroma, template)
            
            # Weight the score slightly by the chord's sparsity
            # This helps prefer simpler chords when scores are close
            chord_complexity = sum(template)
            adjusted_score = score * (1.0 - 0.01 * chord_complexity)
            
            # Store this candidate
            root_name = NOTE_NAMES[root]
            chord_name = root_name if chord_type == "major" else f"{root_name}{chord_type}"
            candidates.append((chord_name, adjusted_score))
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_chord = chord_name
    
    # Adjust confidence threshold based on segment duration
    # Use a lower base threshold for more sensitivity to chord differences
    confidence_threshold = 0.5  # Reduced base threshold for greater sensitivity
    
    if segment_duration:
        # Even for longer segments, keep threshold lower for more sensitivity
        if segment_duration > 1.0:  # For segments longer than 1 second
            confidence_threshold = 0.55
        elif segment_duration > 2.0:  # For segments longer than 2 seconds
            confidence_threshold = 0.6
            
    # Print chord detection details occasionally for debugging
    if random.random() < 0.15:  # Print for ~15% of segments
        print(f"Chord detection details: Best score={best_score:.3f}, Threshold={confidence_threshold:.3f}")
        # Show top 3 chord candidates
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
        print(f"Top candidates: {', '.join([f'{c[0]} ({c[1]:.3f})' for c in sorted_candidates])}")
    
    # Return "N.C." (No Chord) if confidence is too low
    if best_score < confidence_threshold:
        return "N.C."
    
    return best_chord

def separate_sources(audio, sr=SAMPLE_RATE):
    """
    Separate audio into harmonic sources (instruments+bass) and instrumental-only using Demucs.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        
    Returns:
        tuple: (combined_harmonic_audio, instruments_only_audio) as numpy arrays
        - combined_harmonic_audio contains both instruments and bass (no drums, no vocals)
        - instruments_only_audio contains just the instruments without bass
    """
    if not DEMUCS_AVAILABLE:
        print("Demucs not available. Using original audio.")
        return audio, audio
    
    try:
        print("Separating audio sources using Demucs...")
        start_time = time.time()
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
            # Write audio to temp file, handling mono/stereo correctly
            if audio.ndim == 1:
                sf.write(temp_path, audio, sr, format='WAV')
            else:
                sf.write(temp_path, audio.T, sr, format='WAV')
        
        print(f"Audio saved to temporary file: {temp_path}")
        
        # Create a directory for separated sources
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary output directory: {temp_dir}")
            
            # Import the specific separator we need
            from demucs.separate import main as separate_main
            
            # Build arguments for the separator
            original_args = sys.argv
            sys.argv = [
                'demucs',
                '--two-stems=vocals', # We can use vocals/other or drums+bass/other
                '-n', DEMUCS_MODEL,
                '--out', temp_dir,
                temp_path
            ]
            
            # Run the separator
            try:
                separate_main()
            except SystemExit:
                # Demucs calls sys.exit() when done, catch it
                pass
            finally:
                # Restore original args
                sys.argv = original_args
            
            # Load the separated files
            model_folder = f"{DEMUCS_MODEL}"
            track_name = os.path.splitext(os.path.basename(temp_path))[0]
            
            # Try to find the specific stems based on the model's output structure
            # "no_vocals" is used in newer Demucs models for everything except vocals
            no_vocals_path = os.path.join(temp_dir, model_folder, "no_vocals", track_name + ".wav")
            
            # "other" is the instrumental stem in older models
            other_path = os.path.join(temp_dir, model_folder, "other", track_name + ".wav")
            
            # "bass" is the bass stem
            bass_path = os.path.join(temp_dir, model_folder, "bass", track_name + ".wav")
            
            # Load instruments track
            if os.path.exists(no_vocals_path):
                print(f"Loading separated no_vocals track: {no_vocals_path}")
                instruments_and_bass, _ = librosa.load(no_vocals_path, sr=SAMPLE_RATE, mono=True)
                # This already includes bass, so we'll use it as our combined harmonic
                combined_harmonic = instruments_and_bass
                instruments_only = instruments_and_bass  # Will be overridden if bass is available
            elif os.path.exists(other_path):
                print(f"Loading separated instrumental track: {other_path}")
                instruments_only, _ = librosa.load(other_path, sr=SAMPLE_RATE, mono=True)
                combined_harmonic = instruments_only  # Will be combined with bass if available
            else:
                print(f"No separated instrumental file found, using original audio")
                instruments_only = audio
                combined_harmonic = audio
            
            # If bass is available, add it to the harmonic content
            if os.path.exists(bass_path):
                print(f"Loading separated bass track: {bass_path}")
                bass, _ = librosa.load(bass_path, sr=SAMPLE_RATE, mono=True)
                
                # Create a weighted combination of instruments and bass
                # Bass is important for chord detection, but we want to emphasize the harmonic content
                if instruments_only is not combined_harmonic:  # Don't double-add bass if no_vocals already has it
                    combined_harmonic = instruments_only * 0.7 + bass * 0.3
                
            # Cleanup the temporary file
            os.unlink(temp_path)
        
        print(f"Source separation completed successfully")
        elapsed = time.time() - start_time
        print(f"Source separation completed in {elapsed:.1f} seconds")
        
        return combined_harmonic, instruments_only
        
    except Exception as e:
        print(f"Error in source separation: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to original audio")
        return audio, audio

def process_audio_file(audio_path, use_beat_based=False, beats_per_bar=4, use_demucs=True):
    """
    Process an audio file to detect beats, group into segments, and identify chords.
    Each segment (bar or beat) gets one chord assignment.
    
    Args:
        audio_path: Path to the audio file
        use_beat_based: If True, use beat-based segmentation; otherwise use bar-based
        beats_per_bar: Number of beats per bar (for bar-based segmentation)
        use_demucs: If True, use Demucs for source separation
        
    Returns:
        results: Dictionary with BPM and chord information
    """
    try:
        print(f"Processing audio file: {audio_path}")
        
        # Load audio file
        print("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Basic audio info
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio duration: {duration:.2f} seconds")
        
        # Source separation using Demucs
        if use_demucs and DEMUCS_AVAILABLE:
            print("Performing source separation to isolate harmonic content...")
            # First value is the combined harmonic content, second is just instruments
            harmonic_audio, instruments_only = separate_sources(y, sr)
            print("Using combined instrumental and bass audio for chord detection")
        else:
            # Just apply a high-pass filter to remove low frequency noise
            harmonic_audio = librosa.effects.preemphasis(y)
        
        # Detect beats
        print("Detecting beats...")
        beats, beat_times, tempo = detect_beats(y, sr)
        
        if len(beats) == 0:
            print("No beats detected in the audio file after fallback.")
            return {"bpm": 0, "chords": []}
        
        print(f"Detected {len(beats)} beats at tempo {tempo:.1f} BPM")
        
        # Determine segmentation approach (bar-based or beat-based)
        if use_beat_based:
            print("Using beat-based analysis (each beat is a segment)")
            segments = [(int(beats[i]), int(beats[i+1] if i+1 < len(beats) else beats[i] + (beats[i] - beats[i-1]))) 
                        for i in range(len(beats)) if i+1 < len(beats) or i > 0]
            segment_times = [beat_times[i] for i in range(len(beat_times)) if i < len(segments)]
            print(f"Created {len(segments)} beat segments")
        else:
            # Bar-based approach (default)
            print(f"Using {beats_per_bar} beats per segment (bar-based analysis)")
            segments = []
            segment_times = []
            
            for i in range(0, len(beats), beats_per_bar):
                if i + 1 < len(beats):  # Need at least 2 beats to define a segment
                    segment_start = beats[i]
                    segment_start_time = beat_times[i]
                    
                    # Determine segment end
                    if i + beats_per_bar < len(beats):
                        # Full bar available
                        segment_end = beats[i + beats_per_bar]
                    else:
                        # Use remaining beats for final segment
                        # Calculate average beat duration from the last few beats
                        last_beats = beats[-3:] if len(beats) >= 3 else beats[-2:]
                        avg_beat_duration = np.mean(np.diff(last_beats))
                        # Extrapolate to the end of the bar
                        beats_remaining = beats_per_bar - (len(beats) % beats_per_bar)
                        if beats_remaining == beats_per_bar:
                            beats_remaining = 0
                        segment_end = beats[-1] + int(avg_beat_duration * beats_remaining)
                        
                    segments.append((int(segment_start), int(segment_end)))
                    segment_times.append(segment_start_time)
            
            print(f"Created {len(segments)} bar segments")
        
        if not segments:
            print("Could not create segments from beats.")
            return {"bpm": 0, "chords": []}
            
        # Calculate median segment duration for feature extraction
        segment_durations = [(end - start) / sr for start, end in segments]
        median_segment_duration = np.median(segment_durations)
        print(f"Median segment duration: {median_segment_duration:.3f} seconds")
        
        # Initialize results
        results = {
            "bpm": float(tempo),  
            "chords": []
        }
        
        # Process each segment (bar or beat)
        segment_type = "beat" if use_beat_based else "bar"
        print(f"Detecting chords for each {segment_type} segment...")
        
        for i, ((segment_start, segment_end), segment_time) in enumerate(zip(segments, segment_times)):
            try:
                # Get audio segment from the HARMONIC audio
                # Ensure we don't go out of bounds
                if segment_end > len(harmonic_audio):
                    segment_end = len(harmonic_audio)
                    
                segment_audio = harmonic_audio[segment_start:segment_end]
                
                # Determine minimum segment duration based on segmentation type
                min_segment_duration = sr * 0.5 if not use_beat_based else sr * 0.1
                
                # Skip segments that are too short
                if len(segment_audio) < min_segment_duration:
                    print(f"Skipping segment {i+1} - too short: {len(segment_audio)/sr:.3f}s")
                    continue
                
                # Calculate segment duration in seconds
                segment_duration = len(segment_audio) / sr
                
                # Extract chroma features using FFT
                chroma = detect_chroma(segment_audio, sr)
                
                # Identify the chord, passing segment duration for better accuracy
                chord = identify_chord(chroma, segment_duration)
                
                # Calculate start time in milliseconds
                start_time_ms = int(segment_time * 1000)
                
                # Add to results
                results["chords"].append({
                    "start": start_time_ms,
                    "chord": chord
                })
                
                # Keep console output cleaner by only printing every few segments
                display_interval = 4 if not use_beat_based else 8
                if i % display_interval == 0 or i < 10:
                    print(f"{segment_type.title()} {i+1}: {chord} at {start_time_ms}ms (duration: {segment_duration:.3f}s)")
                    
            except Exception as e:
                print(f"Error processing segment {i+1}: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        import traceback
        traceback.print_exc()
        return {"bpm": 0, "chords": []}

def save_results(results, output_path):
    """
    Save results to a JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save the results
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python detect.py <audio_file_path> [options]")
        print("Options:")
        print("  --output=<file_path>      Specify output JSON file path (default: results_fft.json)")
        print("  --beat-based              Use beat-based segmentation (default is bar-based)")
        print("  --beats-per-bar=<num>     Specify beats per bar (default is 4 for 4/4 time)")
        print("  --no-demucs               Disable Demucs source separation")
        sys.exit(1)
    
    # Parse command line arguments
    audio_path = sys.argv[1]
    
    # Parse options
    output_path = OUTPUT_PATH
    use_beat_based = False
    beats_per_bar = 4  # Default to 4/4 time
    use_demucs = True  # Default to using source separation if available
    
    for arg in sys.argv[2:]:
        if arg.startswith("--output="):
            output_path = arg.split("=")[1]
            print(f"Output will be saved to {output_path}")
        elif arg == "--beat-based":
            use_beat_based = True
            print("Using beat-based segmentation")
        elif arg.startswith("--beats-per-bar="):
            try:
                beats_per_bar = int(arg.split("=")[1])
                print(f"Using {beats_per_bar} beats per bar")
            except ValueError:
                print(f"Warning: Invalid beats per bar value. Using default of 4.")
                beats_per_bar = 4
        elif arg == "--no-demucs":
            use_demucs = False
            print("Source separation disabled")
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
    
    # Process audio file
    print("\n--- Starting FFT-based chord detection ---\n")
    results = process_audio_file(audio_path, use_beat_based=use_beat_based, 
                                beats_per_bar=beats_per_bar, use_demucs=use_demucs)
    
    # Post-process chord detection results
    print("\n--- Post-processing chord detection results ---")
    
    # For bar-based analysis, we can skip the time-based filtering 
    # since each segment already represents a meaningful musical unit (a bar)
    if not use_beat_based:
        print("Using bar-based post-processing (skipping time-based filtering)")
        filtered_chords = results["chords"]
    else:
        # For beat-based analysis, we still need to filter rapid changes
        min_duration_ms = 200  # 200ms between chord changes for beat-based
        print(f"Using minimum duration threshold of {min_duration_ms}ms between chord changes")
        filtered_chords = []
        
        # Apply a simple smoothing filter for beat-based analysis
        if results["chords"]:
            prev_chord = None
            chord_buffer = []
            
            for chord_info in results["chords"]:
                if prev_chord is None:
                    # First chord
                    prev_chord = chord_info
                    chord_buffer.append(chord_info)
                else:
                    # Calculate time since previous chord
                    time_diff = chord_info["start"] - prev_chord["start"]
                    
                    if time_diff < min_duration_ms:
                        # Too close to previous - add to buffer for voting
                        chord_buffer.append(chord_info)
                    else:
                        # Enough time has passed - select most common chord from buffer
                        if chord_buffer:
                            # Simple voting: choose the most frequent chord in the buffer
                            chord_counts = {}
                            for c in chord_buffer:
                                chord_counts[c["chord"]] = chord_counts.get(c["chord"], 0) + 1
                            
                            # Find the most common chord
                            most_common = max(chord_counts.items(), key=lambda x: x[1])[0]
                            
                            # Use the timestamp of the first chord in buffer
                            filtered_chords.append({
                                "start": chord_buffer[0]["start"],
                                "chord": most_common
                            })
                        
                        # Reset for next group
                        chord_buffer = [chord_info]
                        prev_chord = chord_info
            
            # Process the last buffer
            if chord_buffer:
                chord_counts = {}
                for c in chord_buffer:
                    chord_counts[c["chord"]] = chord_counts.get(c["chord"], 0) + 1
                
                most_common = max(chord_counts.items(), key=lambda x: x[1])[0]
                filtered_chords.append({
                    "start": chord_buffer[0]["start"],
                    "chord": most_common
                })
    
    # Option to keep all individual bar segments without merging
    print("Showing all individual bar segments without merging...")
    merged_chords = filtered_chords.copy()  # Keep all bars separate
    
    # Calculate and display statistics
    unique_chord_count = len(set([c["chord"] for c in filtered_chords]))
    print(f"Post-processing: {len(results['chords'])} raw segments → {len(filtered_chords)} filtered → {unique_chord_count} unique chord types")
    print(f"Keeping all {len(merged_chords)} individual bar segments separate")
    results["chords"] = merged_chords
    
    # Save results
    save_results(results, output_path)
    
    # Print summary
    print(f"\nDetected {len(results['chords'])} individual bar segments at {results['bpm']:.1f} BPM")
    if len(results["chords"]) > 0:
        print("\nFirst few chords:")
        for i, chord_info in enumerate(results["chords"][:5]):
            # Calculate chord duration if possible
            duration_str = ""
            if i < len(results["chords"]) - 1:
                duration_ms = results["chords"][i+1]["start"] - chord_info["start"]
                duration_str = f"(duration: {duration_ms / 1000:.2f}s)"
            
            print(f"{i+1}. {chord_info['chord']} at {chord_info['start']} ms {duration_str}")
        if len(results["chords"]) > 5:
            print("...")
    
    print(f"\nFull results available in {output_path}")
    print(f"Use 'python midi.py {output_path}' to create a MIDI file of the chord progression")

if __name__ == "__main__":
    main()