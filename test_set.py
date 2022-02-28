"""
QUESTION 2 COMPARISON WITH AVERAGE SPECTRUM
TRY RECORDING AND TESTING AN ADDITIONAL 10 UTTERANCES OF "ONE" AND "TWO"
"""
import numpy as np
import time

from dft import dft
from recordsound import recordsound
from scipy.io.wavfile import write

# RUNS LOOP JUST LIKE IN TRAINING SET FILE
if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency
    num_recordings = 10  # number of recordings for the test set
    digit_recording_list = []

    partial_recs = np.zeros((num_recordings, int(T*fs)))
    print('When prompted to speak, say 1 or 2' + '. \n')
    #USER WILL BE PROMPTED BY THIS LINE, WILL INPUT THEIR VOICE SAYING EITHER DIGIT 1, 10 TIMES, FOLLOWED BY DIGIT 2, 10 TIMES
    for i in range(num_recordings):
        time.sleep(2)
        digit_recorder = recordsound(T, fs)
        spoken_digit = digit_recorder.solve().reshape(int(T*fs))
        partial_recs[i, :] = spoken_digit
    digit_recording_list.append(partial_recs)

    # SAVES THE TEST SET INTO A NPY FILE
    np.save("test_set.npy", partial_recs)

    # SETS AND FILLS WAV FILE WITH RECORDED AUDIO
    # RECORDED AUDIO CONTAINS SPOKEN DIGITS.
    test_set_audio = partial_recs.reshape(T*fs*num_recordings)
    file_name = 'test_set_audio_rec.wav'
    write(file_name, fs, test_set_audio.astype(np.float32))
