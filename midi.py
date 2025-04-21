#!/usr/bin/env python3
"""
midi.py - Converts chord detection JSON to MIDI file

Usage:
    python midi.py results.json [output.mid]

This script takes the JSON output from infer.py and creates a MIDI file 
with the detected chords. The MIDI file will contain chord changes at the 
timestamps specified in the JSON, with each chord played as a block chord.
"""

import json
import sys
import os
from mido import Message, MidiFile, MidiTrack, MetaMessage
import mido
import numpy as np

# MIDI settings
DEFAULT_VELOCITY = 70  # Medium velocity
DEFAULT_PROGRAM = 0    # Piano
TICKS_PER_BEAT = 480   # Standard MIDI resolution

# Mapping of chord names to MIDI notes (relative to C=0)
# Basic mapping from chord name to chord intervals
CHORD_INTERVALS = {
    # Major chords
    '': [0, 4, 7],           # Major (C = C, E, G)
    'maj': [0, 4, 7],        # Major (same as '')
    'M': [0, 4, 7],          # Major (alternate notation)
    
    # Minor chords
    'm': [0, 3, 7],          # Minor (Cm = C, Eb, G)
    'min': [0, 3, 7],        # Minor (alternate notation)
    '-': [0, 3, 7],          # Minor (alternate notation)
    
    # Seventh chords
    '7': [0, 4, 7, 10],      # Dominant 7th (C7 = C, E, G, Bb)
    'maj7': [0, 4, 7, 11],   # Major 7th (Cmaj7 = C, E, G, B)
    'M7': [0, 4, 7, 11],     # Major 7th (alternate)
    'm7': [0, 3, 7, 10],     # Minor 7th (Cm7 = C, Eb, G, Bb)
    'min7': [0, 3, 7, 10],   # Minor 7th (alternate)
    'dim7': [0, 3, 6, 9],    # Diminished 7th (Cdim7 = C, Eb, Gb, A)
    'ø7': [0, 3, 6, 10],     # Half-diminished 7th (Cø7 = C, Eb, Gb, Bb)
    'm7b5': [0, 3, 6, 10],   # Half-diminished 7th (alternate)
    
    # Extended chords
    '9': [0, 4, 7, 10, 14],  # Dominant 9th (C9 = C, E, G, Bb, D)
    'maj9': [0, 4, 7, 11, 14], # Major 9th
    'm9': [0, 3, 7, 10, 14], # Minor 9th
    '11': [0, 4, 7, 10, 14, 17], # 11th chord
    '13': [0, 4, 7, 10, 14, 21], # 13th chord
    
    # Suspended chords
    'sus2': [0, 2, 7],       # Suspended 2nd (Csus2 = C, D, G)
    'sus4': [0, 5, 7],       # Suspended 4th (Csus4 = C, F, G)
    
    # Augmented and diminished chords
    'aug': [0, 4, 8],        # Augmented (Caug = C, E, G#)
    '+': [0, 4, 8],          # Augmented (alternate)
    'dim': [0, 3, 6],        # Diminished (Cdim = C, Eb, Gb)
    'o': [0, 3, 6],          # Diminished (alternate)
    
    # Added tone chords
    'add9': [0, 4, 7, 14],   # Add9 (Cadd9 = C, E, G, D)
    '6': [0, 4, 7, 9],       # 6th (C6 = C, E, G, A)
    'm6': [0, 3, 7, 9],      # Minor 6th (Cm6 = C, Eb, G, A)
}

# Root note mapping
ROOT_NOTES = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def parse_chord(chord_name):
    """
    Parse a chord name and return the MIDI notes for that chord.
    
    Args:
        chord_name: String like "C", "Cm", "G7", etc.
        
    Returns:
        List of MIDI note numbers for that chord
    """
    if chord_name == "N.C." or chord_name == "Unknown" or "Unknown-" in chord_name:
        return []  # No chord
    
    # Extract root and type
    if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
        root = chord_name[:2]
        chord_type = chord_name[2:]
    else:
        root = chord_name[:1]
        chord_type = chord_name[1:]
    
    # Handle root note
    if root not in ROOT_NOTES:
        print(f"Warning: Unknown root note {root} in chord {chord_name}")
        return []
    
    root_note = ROOT_NOTES[root]
    
    # Handle chord type
    if chord_type not in CHORD_INTERVALS:
        # Try to handle some common variations and extensions
        if 'add' in chord_type:
            chord_type = 'add9'  # Simplify to add9
        elif 'sus' in chord_type:
            if '4' in chord_type:
                chord_type = 'sus4'
            else:
                chord_type = 'sus2'
        elif '13' in chord_type:
            chord_type = '13'
        elif '11' in chord_type:
            chord_type = '11'
        elif '9' in chord_type:
            chord_type = '9'
        elif '7' in chord_type:
            if 'maj' in chord_type.lower() or 'M' in chord_type:
                chord_type = 'maj7'
            elif 'm' in chord_type.lower() or '-' in chord_type:
                chord_type = 'm7'
            else:
                chord_type = '7'
        elif 'm' in chord_type.lower() or '-' in chord_type or 'min' in chord_type.lower():
            chord_type = 'm'
        else:
            chord_type = ''  # Default to major
    
    # Get intervals for this chord type
    intervals = CHORD_INTERVALS.get(chord_type, [0, 4, 7])  # Default to major if type not found
    
    # Convert to MIDI notes (Middle C = 60)
    octave = 4  # Middle octave
    midi_notes = [root_note + interval + (octave * 12) for interval in intervals]
    
    return midi_notes

def ms_to_ticks(ms, tempo, ticks_per_beat):
    """
    Convert milliseconds to MIDI ticks.
    
    Args:
        ms: Time in milliseconds
        tempo: Tempo in BPM
        ticks_per_beat: MIDI ticks per beat
        
    Returns:
        Time in MIDI ticks
    """
    beats_per_ms = tempo / (60 * 1000)
    beats = ms * beats_per_ms
    ticks = beats * ticks_per_beat
    return int(ticks)

def create_midi_file(json_data, output_path):
    """
    Create a MIDI file from the chord detection JSON.
    
    Args:
        json_data: The parsed JSON data with chord information
        output_path: Path to save the MIDI file
    """
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    
    # Track 0: Tempo and time signature
    track0 = MidiTrack()
    mid.tracks.append(track0)
    
    # Add tempo (convert BPM to microseconds per beat)
    tempo = json_data.get('bpm', 120)
    if tempo <= 0:
        tempo = 120  # Default to 120 BPM if invalid
    
    # Convert BPM to microseconds per beat
    tempo_us = int(60 * 1000 * 1000 / tempo)
    track0.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    
    # Add time signature (assume 4/4)
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Track 1: Chord progression
    track1 = MidiTrack()
    mid.tracks.append(track1)
    
    # Set the instrument (Piano by default)
    track1.append(Message('program_change', program=DEFAULT_PROGRAM, time=0))
    
    # Get the chord data
    chords = json_data.get('chords', [])
    
    # If no chords, return an empty MIDI file
    if not chords:
        print("Warning: No chords found in the JSON data")
        mid.save(output_path)
        return
    
    # Sort chords by start time (just in case)
    chords = sorted(chords, key=lambda x: x.get('start', 0))
    
    # Previous chord end time in ticks (for calculating delta time)
    prev_end_ticks = 0
    
    # Process each chord
    for i, chord_data in enumerate(chords):
        chord_name = chord_data.get('chord', '')
        start_ms = chord_data.get('start', 0)
        
        # Convert start time to ticks
        start_ticks = ms_to_ticks(start_ms, tempo, TICKS_PER_BEAT)
        
        # Calculate delta time from previous events
        delta_ticks = start_ticks - prev_end_ticks
        if delta_ticks < 0:
            delta_ticks = 0
        
        # Parse the chord to get MIDI notes
        midi_notes = parse_chord(chord_name)
        
        # Skip if no valid notes
        if not midi_notes:
            continue
        
        # Calculate end time (when the next chord starts or a reasonable duration)
        if i < len(chords) - 1:
            next_start_ms = chords[i+1].get('start', 0)
            duration_ms = next_start_ms - start_ms
        else:
            # For the last chord, use a reasonable duration (e.g., 2 seconds)
            duration_ms = 2000
        
        # Convert duration to ticks
        duration_ticks = ms_to_ticks(duration_ms, tempo, TICKS_PER_BEAT)
        
        # Ensure reasonable duration
        if duration_ticks <= 0:
            duration_ticks = TICKS_PER_BEAT  # Default to one beat
        
        # Add note_on events
        for note in midi_notes:
            track1.append(Message('note_on', note=note, velocity=DEFAULT_VELOCITY, time=delta_ticks))
            delta_ticks = 0  # Only the first note has the delta time
        
        # Add note_off events (after the duration)
        for j, note in enumerate(midi_notes):
            if j == len(midi_notes) - 1:
                # Only the last note_off includes the duration
                track1.append(Message('note_off', note=note, velocity=0, time=duration_ticks))
            else:
                track1.append(Message('note_off', note=note, velocity=0, time=0))
        
        # Update previous end time
        prev_end_ticks = start_ticks + duration_ticks
    
    # Add end of track
    track0.append(MetaMessage('end_of_track', time=0))
    track1.append(MetaMessage('end_of_track', time=0))
    
    # Save the MIDI file
    mid.save(output_path)
    print(f"MIDI file saved to {output_path}")

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python midi.py results.json [output.mid]")
        sys.exit(1)
    
    # Get input and output paths
    json_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default output path: replace .json with .mid
        base_name = os.path.splitext(json_path)[0]
        output_path = f"{base_name}.mid"
    
    # Check if input file exists
    if not os.path.exists(json_path):
        print(f"Error: Input file {json_path} not found.")
        sys.exit(1)
    
    # Load JSON data
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
    
    # Create MIDI file
    try:
        create_midi_file(json_data, output_path)
        print(f"Successfully created MIDI file with {len(json_data.get('chords', []))} chords")
    except Exception as e:
        print(f"Error creating MIDI file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()