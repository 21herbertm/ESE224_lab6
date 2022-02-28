"""
@author: Vinicius Lima
This function records spoken digits to create a test set.
"""
import numpy as np
import time

from dft import dft
from recordsound import recordsound
from scipy.io.wavfile import write

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency
    num_recs = 10  # number of recordings for the test set
    digit_recs = []

    partial_recs = np.zeros((num_recs, int(T*fs)))
    print('When prompted to speak, say 1 or 2' + '. \n')
    for i in range(num_recs):
        time.sleep(2)
        digit_recorder = recordsound(T, fs)
        spoken_digit = digit_recorder.solve().reshape(int(T*fs))
        partial_recs[i, :] = spoken_digit
    digit_recs.append(partial_recs)

    # Storing recorded voices
    np.save("test_set.npy", partial_recs)

    # Creating an audio file with the spoken digits
    test_set_audio = partial_recs.reshape(T*fs*num_recs)
    file_name = 'test_set_audio_rec.wav'
    write(file_name, fs, test_set_audio.astype(np.float32))
