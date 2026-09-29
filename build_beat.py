import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly

print("Writing build_beat.py script...")

def load_audio(path, target_sr=44100):
    y, sr = librosa.load(path, sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    return y, sr

def pitch_shift_audio(y, sr, n_steps):
    if n_steps == 0:
        return y
    # Pitch shift stereo channels
    left = librosa.effects.pitch_shift(y[0], sr=sr, n_steps=n_steps)
    right = librosa.effects.pitch_shift(y[1], sr=sr, n_steps=n_steps)
    return np.vstack([left, right])

def change_speed(y, factor):
    if factor == 1.0:
        return y
    left = librosa.effects.time_stretch(y[0], rate=factor)
    right = librosa.effects.time_stretch(y[1], rate=factor)
    return np.vstack([left, right])

def add_sample(buffer, sample, start_sample, gain=1.0, pan=0.0):
    # pan: -1.0 (left) to 1.0 (right)
    if sample.shape[1] == 0:
        return
    end_sample = start_sample + sample.shape[1]
    if start_sample >= buffer.shape[1]:
        return
    if end_sample > buffer.shape[1]:
        sample = sample[:, :buffer.shape[1] - start_sample]
        end_sample = buffer.shape[1]

    # Calculate channel gains for panning
    l_gain = gain * np.cos((pan + 1) * np.pi / 4)
    r_gain = gain * np.sin((pan + 1) * np.pi / 4)

    buffer[0, start_sample:end_sample] += sample[0] * l_gain
    buffer[1, start_sample:end_sample] += sample[1] * r_gain

def soft_clip(x, threshold=0.85):
    # Smooth soft clipper curve
    return np.where(np.abs(x) < threshold, x, np.sign(x) * (threshold + (1 - threshold) * np.tanh((np.abs(x) - threshold) / (1 - threshold))))

def main():
    SR = 44100
    BPM = 150
    BEAT_SEC = 60.0 / BPM
    BAR_SEC = BEAT_SEC * 4
    SIXTEENTH_SEC = BEAT_SEC / 4
    BAR_SAMPLES = int(BAR_SEC * SR)
    SIXTEENTH_SAMPLES = int(SIXTEENTH_SEC * SR)

    TOTAL_BARS = 64
    TOTAL_SAMPLES = TOTAL_BARS * BAR_SAMPLES

    print(f"Total Song Duration: {TOTAL_BARS * BAR_SEC:.2f} seconds ({TOTAL_SAMPLES} samples)")

    master_bus = np.zeros((2, TOTAL_SAMPLES), dtype=np.float32)

    # Load samples
    m1_path = 'Cymatics - 808 Mob Hip-Hop Sample Pack/Melody Loops/Cymatics - 808 Mob Melody Loop 10 - 150 BPM C Min.wav'
    m2_path = 'Cymatics - Diamonds II Hip Hop Sample Pack/Melody Loops/Cymatics - Diamonds Melody Loop 38 - 150 BPM C# Min.wav'
    b808_path = 'Cymatics - 808 Mob Hip-Hop Sample Pack/Drums - One Shots/808s/Cymatics - 808 Mob 808 1 - C.wav'
    kick_path = 'Cymatics - 808 Mob Hip-Hop Sample Pack/Drums - One Shots/Kicks/Cymatics - 808 Mob Kick 1 - F#.wav'
    clap_path = 'Cymatics - Diamonds II Hip Hop Sample Pack/Drums - One Shots/Claps/Cymatics - Diamonds Clap 1.wav'
    snare_path = 'Cymatics - 808 Mob Hip-Hop Sample Pack/Drums - One Shots/Snares/Cymatics - 808 Mob Snare 1 - D#.wav'
    hh_path = 'Cymatics - Diamonds II Hip Hop Sample Pack/Drums - One Shots/Cymbals/Hihats - Closed/Cymatics - Diamonds Closed Hihat 1.wav'
    oh_path = 'Cymatics - 808 Mob Hip-Hop Sample Pack/Drums - One Shots/Cymbals/Hihats - Open/Cymatics - 808 Mob Open Hihat 1.wav'
    vox_path = 'Cymatics - Diamonds II Hip Hop Sample Pack/Vocals/Cymatics - Diamonds Vocal 1.wav'
    fx_siren_path = 'Cymatics - 808 Mob Hip-Hop Sample Pack/FX/Cymatics - 808 Mob FX 12 - Kill Bill Siren 2.wav'
    fx_alarm_path = 'Cymatics - Diamonds II Hip Hop Sample Pack/FX/Various FX/Cymatics - Diamonds FX 10 - Damaged Alarm.wav'

    print("Loading and preparing audio files...")
    m1, _ = load_audio(m1_path, SR)
    m2, _ = load_audio(m2_path, SR)
    b808, _ = load_audio(b808_path, SR)
    kick, _ = load_audio(kick_path, SR)
    clap, _ = load_audio(clap_path, SR)
    snare, _ = load_audio(snare_path, SR)
    hh, _ = load_audio(hh_path, SR)
    oh, _ = load_audio(oh_path, SR)
    vox, _ = load_audio(vox_path, SR)
    siren, _ = load_audio(fx_siren_path, SR)
    alarm, _ = load_audio(fx_alarm_path, SR)

    # Key alignment:
    # m1 is C Min. Pitch shift +1 semitone to C# Min.
    print("Pitch shifting Melody 1 (+1 semitone to C# Min)...")
    m1_cs = pitch_shift_audio(m1, SR, 1)

    # 808 sample is C. Shift +1 semitone to C# as base root.
    b808_root = pitch_shift_audio(b808, SR, 1)

    # Pre-calculate 808 pitch variations in C# Minor scale (C#, E, F#, G#, B)
    # Pitch offsets relative to C#:
    # C# = +0, E = +3, F# = +5, G# = +7, B = +10, High C# = +12
    print("Pre-generating 808 pitch variations...")
    notes_808 = {}
    for semitone in [0, 3, 5, 7, 10, 12, -2]:
        notes_808[semitone] = pitch_shift_audio(b808_root, SR, semitone)

    # Pre-generate hihat pitched rolls
    hh_notes = {}
    for pitch in [-4, -2, 0, 2, 4, 7]:
        hh_notes[pitch] = pitch_shift_audio(hh, SR, pitch)

    # Structure arrangement:
    # Bars 0-7 (1-8): Intro
    # Bars 8-23 (9-24): Drop 1 / Chorus 1 (16 bars)
    # Bars 24-39 (25-40): Verse 1 (16 bars)
    # Bars 40-55 (41-56): Drop 2 / Chorus 2 (16 bars)
    # Bars 56-63 (57-64): Outro (8 bars)

    print("Arranging Melodies...")
    # Melody 1 plays throughout intro, drops, verse, outro
    m1_length = m1_cs.shape[1]
    for bar in range(TOTAL_BARS):
        start_samp = bar * BAR_SAMPLES
        # Calculate offset in 16-bar loop
        loop_pos = (bar % 16) * BAR_SAMPLES
        chunk = m1_cs[:, loop_pos:loop_pos + BAR_SAMPLES]

        # Gain staging per section
        if bar < 8: # Intro
            gain = 0.5
        elif 8 <= bar < 24 or 40 <= bar < 56: # Drops
            gain = 0.65
        elif 24 <= bar < 40: # Verse
            gain = 0.55
        else: # Outro
            gain = 0.45 * (1.0 - (bar - 56) / 8.0) # Fade out

        add_sample(master_bus, chunk, start_samp, gain=gain)

    # Melody 2 (Secondary synth layer) enters on Drops and second half of Verse
    m2_length = m2.shape[1]
    for bar in range(TOTAL_BARS):
        if (8 <= bar < 24) or (32 <= bar < 40) or (40 <= bar < 56): # Drops + 2nd half Verse
            start_samp = bar * BAR_SAMPLES
            loop_pos = (bar % 16) * BAR_SAMPLES
            chunk = m2[:, loop_pos:loop_pos + BAR_SAMPLES]
            gain = 0.40 if (8 <= bar < 24 or 40 <= bar < 56) else 0.25
            add_sample(master_bus, chunk, start_samp, gain=gain, pan=0.2)

    print("Arranging FX and Vocal Chops...")
    # FX Siren on Intro and Drop transitions
    add_sample(master_bus, siren, 0, gain=0.5) # Intro start
    add_sample(master_bus, siren, 7 * BAR_SAMPLES + int(2 * BEAT_SEC * SR), gain=0.6) # Drop 1 transition
    add_sample(master_bus, siren, 39 * BAR_SAMPLES + int(2 * BEAT_SEC * SR), gain=0.6) # Drop 2 transition

    # Damaged Alarm on drop impact bars
    for drop_bar in [8, 16, 40, 48]:
        add_sample(master_bus, alarm, drop_bar * BAR_SAMPLES, gain=0.45, pan=-0.3)

    # Vocal chop on beat 1 of drop bars & verse accent
    for bar in [8, 12, 16, 20, 28, 36, 40, 44, 48, 52]:
        add_sample(master_bus, vox, bar * BAR_SAMPLES, gain=0.4, pan=0.1)

    print("Arranging Drums & 808 Bass...")

    # 808 Bass patterns (16th note step index per bar)
    # Pattern A (Rage Bounce - Drop 1 & Drop 2)
    # Steps: 0 (C#), 6 (C#), 10 (E), 12 (F#), 14 (G#)
    drop_808_pattern = [
        # (step_offset, pitch_semitone, duration_sixteenths)
        (0, 0, 5),
        (6, 0, 3),
        (10, 3, 2),
        (12, 5, 2),
        (14, 7, 2),
    ]
    drop_808_pattern_b = [
        (0, 0, 5),
        (6, 0, 3),
        (10, 10, 2),
        (12, 7, 2),
        (14, 12, 2), # Slide up high
    ]

    # Verse 808 pattern (More space)
    verse_808_pattern = [
        (0, 0, 6),
        (8, -2, 4),
        (12, 0, 4),
    ]

    # Kick pattern
    drop_kick_pattern = [0, 6, 10, 12] # Hits with 808 for maximum punch
    verse_kick_pattern = [0, 10]

    # Clap on step 4 and 12 (beats 2 and 4)
    clap_pattern = [4, 12]

    # Snare ghost rolls / fills
    snare_fill_pattern = [14, 15]

    # Generate Hi-Hat rolls per bar
    for bar in range(TOTAL_BARS):
        bar_start = bar * BAR_SAMPLES

        is_drop = (8 <= bar < 24) or (40 <= bar < 56)
        is_verse = (24 <= bar < 40)

        if not (is_drop or is_verse):
            continue # Intro and outro have no heavy drums

        # Claps
        for step in clap_pattern:
            add_sample(master_bus, clap, bar_start + step * SIXTEENTH_SAMPLES, gain=0.75)

        # Snares (verse and drop fills)
        if is_drop or (is_verse and bar >= 32):
            if bar % 2 == 1:
                for step in snare_fill_pattern:
                    add_sample(master_bus, snare, bar_start + step * SIXTEENTH_SAMPLES, gain=0.5, pan=-0.1)

        # Open Hats
        if is_drop:
            for step in [2, 10]: # Off-beat open hats
                add_sample(master_bus, oh, bar_start + step * SIXTEENTH_SAMPLES, gain=0.45, pan=0.25)

        # Kicks
        k_pat = drop_kick_pattern if is_drop else verse_kick_pattern
        if is_drop or (is_verse and bar >= 28):
            for step in k_pat:
                add_sample(master_bus, kick, bar_start + step * SIXTEENTH_SAMPLES, gain=0.85)

        # 808 Bass
        if is_drop:
            pat = drop_808_pattern_b if (bar % 4 == 3) else drop_808_pattern
            for step_offset, pitch, dur in pat:
                samp = notes_808[pitch]
                add_sample(master_bus, samp, bar_start + step_offset * SIXTEENTH_SAMPLES, gain=0.90)
        elif is_verse:
            for step_offset, pitch, dur in verse_808_pattern:
                samp = notes_808[pitch]
                add_sample(master_bus, samp, bar_start + step_offset * SIXTEENTH_SAMPLES, gain=0.80)

        # Rolling Hi-Hats (Yeat signature)
        # 16 steps per bar
        for step in range(16):
            step_time = bar_start + step * SIXTEENTH_SAMPLES

            # Fast rolls on specific steps
            if is_drop and (step in [7, 15]): # 32nd note roll (2 hits in 1 sixteenth)
                for r in range(2):
                    t = step_time + int(r * SIXTEENTH_SAMPLES / 2)
                    pitch = 2 if r == 1 else 0
                    add_sample(master_bus, hh_notes[pitch], t, gain=0.55, pan=-0.15)
            elif is_drop and (step == 11 and bar % 2 == 1): # Triplet / 64th fast drop roll (3 hits)
                for r in range(3):
                    t = step_time + int(r * SIXTEENTH_SAMPLES / 3)
                    pitch = 4 - r * 2
                    add_sample(master_bus, hh_notes[pitch], t, gain=0.50, pan=0.15)
            else:
                # Standard 8th/16th hihat
                if step % 2 == 0 or is_drop:
                    gain = 0.60 if step % 4 == 0 else 0.45
                    add_sample(master_bus, hh_notes[0], step_time, gain=gain, pan=-0.1)

    print("Applying mixing and mastering soft clipping...")
    # Soft clipping to give Yeat style saturation and prevent hard digital clipping
    master_bus[0] = soft_clip(master_bus[0] * 1.15, threshold=0.80)
    master_bus[1] = soft_clip(master_bus[1] * 1.15, threshold=0.80)

    # Master normalization to -0.3 dB peak
    max_peak = np.max(np.abs(master_bus))
    target_peak = 0.96 # ~ -0.35 dBFS
    if max_peak > 0:
        master_bus = master_bus * (target_peak / max_peak)

    os.makedirs('songs', exist_ok=True)
    out_path = 'songs/yeat_type_beat.wav'
    print(f"Writing rendered beat to {out_path}...")
    sf.write(out_path, master_bus.T, SR, subtype='PCM_16')
    print("Done!")

if __name__ == '__main__':
    main()
