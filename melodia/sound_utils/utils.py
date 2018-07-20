import numpy as np
import os
from pydub import AudioSegment
from subprocess import call as cmd
from subprocess import DEVNULL
from subprocess import Popen, PIPE, STDOUT
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
def mp3_to_wav(in_path, out_path):    
    cmd(['C:/Program Files/ffmpeg/bin/ffmpeg', '-i', in_path, out_path])
    # sound = AudioSegment.from_mp3("D:/seq-model/seq-model/melodia/sound_utils/i_see_fire")
    # sound.export(out_path, format="wav")
def extract_melody(in_path, out_path = "out.txt", vamp="D:/vamp-plugin/vamp"):
    if os.path.isfile(out_path) == False:
        shell_command = "{} -s mtg-melodia:melodia {} -o {}".format(vamp, in_path, out_path)
        cmd(shell_command.split(),stdout=DEVNULL, stderr=STDOUT)

    melody =[]
    with open(out_path,"r") as f:
        lines = f.readlines()
    for line in lines:
        melody.append(float(line.split(" ")[1]))
    return np.insert(melody, 0, [0]*8)
def hz2midi(hz):
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[idx] = 0

    midi = np.round(midi)
    return midi
def midi_to_notes(midi, fs, hop, smooth, minduration):
    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi
    print(len(midi),len(midi_filt))
    notes = []
    p_prev = 0
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hop / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p
    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes
def save_midi(outfile, notes, tempo):

    track = 0
    time = 0
    midifile = MIDIFile(1)

    # Add track name and tempo.
    midifile.addTrackName(track, time, "MIDI TRACK")
    midifile.addTempo(track, time, tempo)

    channel = 0
    volume = 100

    for note in notes:
        onset = note[0] * (tempo/60.)
        duration = note[1] * (tempo/60.)
        # duration = 1
        pitch = note[2]
        midifile.addNote(track, channel, np.round(pitch).astype(int), onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()
    # call(["dir"])

# in_file = "i_see_fire"
in_file = "cheap_thrills"
# mp3_to_wav(in_file+".mp3", in_file+".wav")
minduration = 0.1
smooth = 0.25
bpm = 120
fs = 44100
hop = 128

# Step for melodia 
# 1. convert mp3 to wav
# 2. run vamp plugin

melody = extract_melody(in_file+".wav",in_file+".txt")

# convert f0 to midi notes
midi_pitch = hz2midi(melody)

# segment sequence into individual midi notes
notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

# save note sequence to a midi file
save_midi(in_file+".mid", notes, bpm)