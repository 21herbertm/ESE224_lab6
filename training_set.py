"""
TRAINING SET CLASS BUILT FOR QUESTION #1

1 Acquire and process training sets. Acquire and store N = 10 recordings for each of the two digits “one” and “two.”
Compute the respective
DFTs and normalize them so that they have unit energy
1). You can use the provided record sound() class. If you do, please
remember to check help (record sound) before using it.
Having acquired and processed the training sets Y and Z, we acquire a
new signal x that contains the utterance of either the word “one” or the 3
word “two.” We want to (correctly) identify which of these words was
spoken. To do so, we compare the magnitude of the DFT X = F(x)—
which we assume has been normalized to have unit energy—with the
magnitudes of the DFTs Yi and Zi
that were stored in the training sets.
There are different choices to make this comparison. We will try two of
them in this lab. It will probably save you a lot of time to record all the
digits once and store them in a .npy file for future use (i.e., so you don’t
have to re-record them every time you run your code). This can be done
by using the function numpy.save(). In any case, you can ask your TAs
for help on storing the recorded voices. It would also help, if the script
for recording voices and storing them is in a separate file (although, it
would be useful that every script for every exercise is in a separate file)


"""
import numpy as np
import time

from dft import dft
from recordsound import recordsound

# RUNS LOOP
if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency
    num_recordings_per_digit = 10  # number of recordings for each digit
    digits = [1, 2]  # digits to be recorded
    digit_recorded_list = []

    # LOOPS THROUGH ALL DIGITS IN THE LIST GIVEN, INCLUDING JUST 1 AND 2
    # RECORDS SOUND, 10 TIMES PER DIGIT
    for digit in digits:
        # The numpy.zeros() function returns a new array of given shape and type, with zeros.
        # b = geek.zeros(2, dtype = int)
        # Matrix b :
        #  [0 0]

        # GIVEN A 1 BY 10 MATRIX FILLED WITH 0'S
        partial_recs = np.zeros((num_recordings_per_digit, int(T*fs)))
        print('When prompted to speak, say ' + str(digit) + '. \n')
        # LOOP THROUGH 10 TIMES
        for i in range(num_recordings_per_digit):
            time.sleep(2)
            # EACH TIME SETS DIGIT RECORDER TO THE RECORD SOUND WITH PARAMETERS OF SAMPLING FREQ AND TIME
            digit_recorded = recordsound(T, fs)
            spoken_digit = digit_recorded.solve().reshape(int(T*fs))
            partial_recs[i, :] = spoken_digit
        digit_recorded_list.append(partial_recs)

    # Storing recorded voices
    np.save("recorded_digits.npy",digit_recorded_list)

    # Computing the DFTs
    digit_recorded_list = np.load("recorded_digits.npy")
    digits = [1, 2]
    num_digits = len(digit_recs)
    num_recordings_per_digit, N = digit_recorded_list[0].shape
    fs = 8000
    DFTs = []
    DFTs_c = []

# LOOPS FOR ALL dIGITS INSIDE OF THE LIST AT CURRENT TIME
    for digit_rec in digit_recorded_list:
        DFTs_aux = np.zeros((num_recordings_per_digit, N), dtype=np.complex_)
        DFTs_c_aux = np.zeros((num_recordings_per_digit, N), dtype=np.complex_)
        for i in range(num_recordings_per_digit):
            rec_i = digit_rec[i, :]
            # We can use the norm of the ith signal to normalize its DFT
            energy_rec_i = np.linalg.norm(rec_i)
# EX: linalg.norm(x, ord=None, axis=None, keepdims=False)
"""
This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms 
(described below), depending on the value of the ord parameter.

Parameters
xarray_like
Input array. If axis is None, x must be 1-D or 2-D, unless ord is None. If both axis and ord are None, the 2-norm of x.
ravel will be returned.

ord{non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
Order of the norm (see table under Notes). inf means numpy’s inf object. The default is None.

axis{None, int, 2-tuple of ints}, optional.
If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a 2-tuple, it 
specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. If axis is None then either 
a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The default is None.

"""
# CALL TO THE DFT FILE
            rec_i /= energy_rec_i
            DFT_rec_i = dft(rec_i, fs)
            [_, X, _, X_c] = DFT_rec_i.solve3()
            DFTs_aux[i, :] = X 
            DFTs_c_aux[i, :] = X_c
        DFTs.append(DFTs_aux)
        DFTs_c.append(DFTs_c_aux) 

    # SAVES TIME BY RECORDING THE DIGITS IN NPY FILE
    np.save("spoken_digits_DFTs.npy", DFTs)
    np.save("spoken_digits_DFTs_c.npy", DFTs_c)

    


            