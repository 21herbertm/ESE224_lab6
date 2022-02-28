"""
@author: Vinicius Lima
This function records spoken digits and computes the (normalized) DFT of each sample.
"""
import numpy as np
import time

from dft import dft
from recordsound import recordsound

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency
    num_recs = 10  # number of recordings for each digit
    digits = [1, 2]  # digits to be recorded
    digit_recs = []

    for digit in digits:
        partial_recs = np.zeros((num_recs, int(T*fs)))
        print('When prompted to speak, say ' + str(digit) + '. \n')
        for i in range(num_recs):
            time.sleep(2)
            digit_recorder = recordsound(T, fs)
            spoken_digit = digit_recorder.solve().reshape(int(T*fs))
            partial_recs[i, :] = spoken_digit
        digit_recs.append(partial_recs)

    # Storing recorded voices
    np.save("recorded_digits.npy", digit_recs)

    # Computing the DFTs
    digit_recs = np.load("recorded_digits.npy")
    digits = [1, 2]
    num_digits = len(digit_recs)
    num_recs, N = digit_recs[0].shape 
    fs = 8000
    DFTs = []
    DFTs_c = []

    for digit_rec in digit_recs:
        DFTs_aux = np.zeros((num_recs, N), dtype=np.complex_)
        DFTs_c_aux = np.zeros((num_recs, N), dtype=np.complex_)
        for i in range(num_recs):
            rec_i = digit_rec[i, :]
            # We can use the norm of the ith signal to normalize its DFT
            energy_rec_i = np.linalg.norm(rec_i)
            rec_i /= energy_rec_i
            DFT_rec_i = dft(rec_i, fs)
            [_, X, _, X_c] = DFT_rec_i.solve3()
            DFTs_aux[i, :] = X 
            DFTs_c_aux[i, :] = X_c
        DFTs.append(DFTs_aux)
        DFTs_c.append(DFTs_c_aux) 

    # Storing DFTs
    np.save("spoken_digits_DFTs.npy", DFTs)
    np.save("spoken_digits_DFTs_c.npy", DFTs_c)

    


            