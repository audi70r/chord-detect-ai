import numpy as np
from pydub.generators import Sine
from pydub import AudioSegment
import sqlite3
import os
import random
from pychord import Chord

# Parameters for the audio
SAMPLE_RATE = 44100

# Output directory
OUTPUT_DIR = "chord_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect("chords.db")
cursor = conn.cursor()

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    chord_name TEXT
)
""")
conn.commit()

# Generate all chords: roots and types, including 9th and 11th chords
ALL_CHORDS = []
CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TYPES = ["", "m", "7", "m7", "maj7", "dim", "aug", "add9", "9", "11"]

for root in CHORD_ROOTS:
    for ctype in CHORD_TYPES:
        ALL_CHORDS.append(f"{root}{ctype}")


# Function to generate a single chord audio segment
def generate_chord_segment(notes, duration_ms, sample_rate):
    if not notes:
        return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    
    # Create more realistic timbres by using different wave shapes
    # Choose a timbre type for this chord
    timbre = random.choice(['sine', 'piano-like', 'guitar-like', 'organ-like', 'synth'])
    
    combined = None
    for note in notes:
        freq = note_to_freq(note)
        
        try:
            if timbre == 'sine':
                # Pure sine wave
                wave = Sine(freq, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
                wave = wave.apply_gain(-12)  # Avoid clipping
                
            elif timbre == 'piano-like':
                # Piano-like with harmonics and decay
                fundamental = Sine(freq, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
                # Add harmonics
                first_harmonic = Sine(freq * 2, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(-12)
                second_harmonic = Sine(freq * 3, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(-18)
                
                # Combine harmonics
                wave = fundamental.apply_gain(-6)
                wave = wave.overlay(first_harmonic)
                wave = wave.overlay(second_harmonic)
                
                # Create decay envelope (simplified)
                wave = wave.fade_out(int(duration_ms * 0.8))
                
            elif timbre == 'guitar-like':
                # Guitar-like with harmonics and faster decay
                fundamental = Sine(freq, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
                # Add harmonics with specific ratios
                first_harmonic = Sine(freq * 2, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(-8)
                third_harmonic = Sine(freq * 3, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(-16)
                
                # Combine harmonics
                wave = fundamental.apply_gain(-6)
                wave = wave.overlay(first_harmonic)
                wave = wave.overlay(third_harmonic)
                
                # Create faster decay
                wave = wave.fade_out(int(duration_ms * 0.5))
                
            elif timbre == 'organ-like':
                # Organ-like with steady harmonics
                fundamental = Sine(freq, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
                # Add harmonics
                first_harmonic = Sine(freq * 2, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(-6)
                fifth_harmonic = Sine(freq * 5, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(-12)
                
                # Combine harmonics
                wave = fundamental.apply_gain(-6)
                wave = wave.overlay(first_harmonic)
                wave = wave.overlay(fifth_harmonic)
                
            else:  # synth
                # Synth-like with rich harmonics
                fundamental = Sine(freq, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
                
                # Create a few harmonics
                harmonics = []
                for i in range(2, 6):
                    harmonic_gain = -6 * i  # Higher harmonics get progressively quieter
                    harmonic = Sine(freq * i, sample_rate=sample_rate).to_audio_segment(duration=duration_ms).apply_gain(harmonic_gain)
                    harmonics.append(harmonic)
                
                # Combine all harmonics
                wave = fundamental.apply_gain(-6)
                for harmonic in harmonics:
                    wave = wave.overlay(harmonic)
            
            # Add the note to the combined segment
            if combined is None:
                combined = wave
            else:
                combined = combined.overlay(wave)
                
        except Exception as e:
            print(f"Error generating note {note} with frequency {freq}: {e}")
            continue
    
    # If we couldn't generate any notes successfully
    if combined is None:
        # Just create a simple sine wave as fallback
        base_freq = 440  # A4
        combined = Sine(base_freq, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
    
    return combined


# Function to convert MIDI note number to frequency
def note_to_freq(note):
    A4_freq = 440.0
    A4_note = 69  # MIDI note number for A4
    semitones = note - A4_note
    return A4_freq * (2 ** (semitones / 12.0))


# Function to add noise
def add_noise(segment, noise_level=0.01):
    # Get original samples and convert to float for processing
    samples = np.array(segment.get_array_of_samples())
    original_dtype = samples.dtype
    
    # Convert to float32 for processing
    samples = samples.astype(np.float32)
    
    # Determine max value based on original dtype
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
    else:
        max_val = 1.0  # For floating point, assume normalized audio
        
    # Generate noise and add it
    noise = np.random.normal(0, noise_level * max_val, samples.shape).astype(np.float32)
    noisy_samples = samples + noise
    
    # Clip to original range
    if np.issubdtype(original_dtype, np.integer):
        min_val = np.iinfo(original_dtype).min
        max_val = np.iinfo(original_dtype).max
    else:
        min_val = -1.0
        max_val = 1.0
        
    noisy_samples = np.clip(noisy_samples, min_val, max_val)
    
    # Convert back to original dtype
    noisy_samples = noisy_samples.astype(original_dtype)
    
    # Create new segment
    noisy_segment = segment._spawn(noisy_samples.tobytes())
    return noisy_segment


# Function to add transients
def add_transient(segment, attack_ms=10):
    try:
        # Get a random attack time between 5-20ms
        attack_ms = random.randint(5, 20)
        
        # Create multiple transient types for diversity
        transient_type = random.choice(['click', 'thump', 'noise', 'complex'])
        
        if transient_type == 'click':
            # Short, high-frequency click
            freq = random.randint(800, 2000)
            gain = random.uniform(-6, -2)
            transient_wave = Sine(freq, sample_rate=segment.frame_rate).to_audio_segment(duration=attack_ms).apply_gain(gain)
        
        elif transient_type == 'thump':
            # Low frequency thump
            freq = random.randint(40, 120)
            gain = random.uniform(-8, -3)
            transient_wave = Sine(freq, sample_rate=segment.frame_rate).to_audio_segment(duration=attack_ms*2).apply_gain(gain)
        
        elif transient_type == 'noise':
            # White noise burst
            silent = AudioSegment.silent(duration=attack_ms, frame_rate=segment.frame_rate)
            samples = np.array(silent.get_array_of_samples())
            # Use int16 max values to ensure compatibility
            noise = np.random.normal(0, 5000, samples.shape).astype(samples.dtype)
            noise_segment = silent._spawn(noise.tobytes())
            gain = random.uniform(-12, -6)
            transient_wave = noise_segment.apply_gain(gain)
        
        else:  # complex
            # Combination of different transients
            freq1 = random.randint(100, 300)
            freq2 = random.randint(500, 1200)
            wave1 = Sine(freq1, sample_rate=segment.frame_rate).to_audio_segment(duration=attack_ms*2).apply_gain(-5)
            wave2 = Sine(freq2, sample_rate=segment.frame_rate).to_audio_segment(duration=attack_ms).apply_gain(-8)
            transient_wave = wave1.overlay(wave2)
        
        # Add the transient to the beginning of the segment
        combined = transient_wave + segment[attack_ms:]
        
        return combined
        
    except Exception as e:
        print(f"Error adding transient: {e}")
        # If there's an error, just return the original segment
        return segment


# Generate chord segments, add noise and transients, and save them
for chord_name in ALL_CHORDS:
    chord = Chord(chord_name)
    
    # Let's create a simpler, more direct approach to generate chord notes
    # Create a map of chord components to MIDI notes (C4 = middle C = 60)
    note_to_midi = {
        "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
        "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67,
        "G#": 68, "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
    }
    
    # Get root and components
    root = chord_name[0]
    if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
        root += chord_name[1]
    
    # Choose a base octave
    base_octave = random.choice([3, 4, 5])
    octave_shift = (base_octave - 4) * 12  # Middle C (C4) is MIDI note 60
    
    # Get the root note MIDI number with octave shift
    if root in note_to_midi:
        root_midi = note_to_midi[root] + octave_shift
    else:
        # Default to C if we can't parse the root
        root_midi = 60 + octave_shift
    
    # Get all chord components as semitone intervals from root
    components = chord.components()
    
    # Convert component intervals to MIDI notes
    midi_notes = []
    root_index = None
    
    # Find where the root is in the components
    for i, comp in enumerate(components):
        if comp == root:
            root_index = i
            break
    
    if root_index is None:
        # If root not found, add it manually
        midi_notes.append(root_midi)
    
    # Add all components
    for comp in components:
        # Map component name to MIDI note number
        if comp in note_to_midi:
            comp_midi = note_to_midi[comp]
            # Adjust to be in the same octave as root or higher
            while comp_midi < root_midi:
                comp_midi += 12
            # Don't add components that are more than an octave higher
            if comp_midi <= root_midi + 12:
                midi_notes.append(comp_midi)
    
    # Ensure we have at least the root note
    if not midi_notes:
        midi_notes.append(root_midi)
    
    # Add octave of root for more stability in chords
    if len(midi_notes) < 3:
        midi_notes.append(root_midi + 12)
    
    # We'll create multiple variations for each chord
    original_midi_notes = midi_notes.copy()
    variations = []
    
    # Create 5 different octave positions (from very low to very high)
    for octave_shift in [-24, -12, 0, 12, 24]:
        shifted_notes = [note + octave_shift for note in original_midi_notes]
        # Only use if all notes are in a reasonable MIDI range (0-127)
        if all(0 <= note <= 127 for note in shifted_notes):
            variations.append(shifted_notes)
    
    # Create different inversions of the chord
    if len(original_midi_notes) >= 3:
        # First inversion - move root to the top
        first_inv = original_midi_notes.copy()
        root = first_inv[0]
        first_inv.remove(root)
        first_inv.append(root + 12)
        variations.append(first_inv)
        
        # Second inversion - move root and second note to the top
        if len(original_midi_notes) >= 3:
            second_inv = original_midi_notes.copy()
            root = second_inv[0]
            second = second_inv[1] 
            second_inv.remove(root)
            second_inv.remove(second)
            second_inv.append(root + 12)
            second_inv.append(second + 12)
            variations.append(second_inv)
    
    # Add different voicing types
    voicing_types = ['drop2', 'spread', 'add_octave', 'omit_note', 'cluster', 'wide']
    
    for voicing_type in voicing_types:
        notes = original_midi_notes.copy()
        
        if voicing_type == 'drop2' and len(notes) >= 4:
            # Drop 2 voicing: drop the second highest note by an octave
            notes.sort()
            second_highest = notes[-2]
            notes.remove(second_highest)
            notes.append(second_highest - 12)
            variations.append(notes)
            
        elif voicing_type == 'spread' and len(notes) >= 3:
            # Spread voicing: spread the notes across octaves
            notes.sort()
            for i in range(1, len(notes), 2):
                if i < len(notes):
                    notes[i] += 12
            variations.append(notes)
                    
        elif voicing_type == 'add_octave' and len(notes) >= 2:
            # Add octave doubling of some notes
            notes.sort()
            note_to_double = random.choice(notes)
            notes_copy = notes.copy()
            notes_copy.append(note_to_double + 12)
            variations.append(notes_copy)
            
        elif voicing_type == 'omit_note' and len(notes) >= 4:
            # Omit a note (not the root)
            notes.sort()
            note_to_omit = random.choice(notes[1:])
            notes_copy = notes.copy()
            notes_copy.remove(note_to_omit)
            variations.append(notes_copy)
            
        elif voicing_type == 'cluster' and len(notes) >= 3:
            # Create a cluster voicing (all notes within one octave)
            notes.sort()
            cluster = [notes[0]]
            for note in notes[1:]:
                # Keep notes in the same octave
                while note - cluster[0] >= 12:
                    note -= 12
                cluster.append(note)
            variations.append(sorted(cluster))
            
        elif voicing_type == 'wide' and len(notes) >= 3:
            # Create a wide voicing (spread notes across multiple octaves)
            notes.sort()
            wide = [notes[0]]
            for i, note in enumerate(notes[1:]):
                # Space notes apart by octaves
                wide.append(note + (i * 12))
            # Only add if all notes are in a reasonable MIDI range
            if all(0 <= note <= 127 for note in wide):
                variations.append(wide)
    
    # Add the original notes as a variation too
    variations.append(original_midi_notes)
    
    # Debug print
    print(f"Chord: {chord_name}, Generated {len(variations)} variations")
    
    # Generate 2 examples for each variation
    for variation_idx, midi_notes in enumerate(variations):
        # Sort the notes for clean playback
        midi_notes.sort()
        
        # Skip if empty
        if not midi_notes:
            continue
            
        print(f"  Variation {variation_idx}: {midi_notes}")
        
        for i in range(3):  # Generate 3 examples per variation
            # Generate a longer duration between 2 and 4 seconds
            segment_duration_sec = random.uniform(2.0, 4.0)
            segment_duration_ms = int(segment_duration_sec * 1000)

            # Generate the chord segment
            segment = generate_chord_segment(midi_notes, segment_duration_ms, SAMPLE_RATE)
            if segment:
                segment = add_noise(segment, noise_level=0.05)
                segment = add_transient(segment, attack_ms=10)

            # Create a descriptive name for the variation
            variation_type = ""
            if variation_idx < 5:
                # First 5 variations are octave shifts
                octave_names = ["very_low", "low", "mid", "high", "very_high"]
                if variation_idx < len(octave_names):
                    variation_type = f"octave_{octave_names[variation_idx]}"
            elif variation_idx < 7:
                # Next 2 are inversions
                inversion_names = ["first_inv", "second_inv"]
                variation_type = inversion_names[variation_idx - 5]
            elif variation_idx < 13:
                # Next 6 are voicing types
                voicing_names = ["drop2", "spread", "add_octave", "omit_note", "cluster", "wide"]
                variation_type = f"voicing_{voicing_names[variation_idx - 7]}"
            else:
                # Original voicing
                variation_type = "original"
            
            file_name = f"{chord_name}_{variation_type}_example_{i}.wav"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            
            if segment:
                segment.export(file_path, format="wav")
                
                try:
                    cursor.execute("INSERT INTO segments (file_name, chord_name) VALUES (?, ?)", (file_name, chord_name))
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting into database: {e}")
            else:
                print(f"Error generating segment for chord: {chord_name}")

# Close the database connection
conn.close()

print(f"Chord segments generated and metadata saved in {OUTPUT_DIR} and chords.db")