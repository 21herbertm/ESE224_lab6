"""
MELANIE HERBERT, ALINA HO
ESE 224 LAB 6

AVERAGE_COMP FILE

QUESTION 2 COMPARISON WITH AVERAGE SPECTRUM

For each of the training sets
define the average spectral magnitudes
(1)
Further define the inner product p(X,Y) between the spectra of any two
signals X and Y as the inner product between their absolute values,

(2)
Compare the inner product p(X,Y¯) between the unknown spectrum X
and the average spectrum Y¯ with the inner product p(X, Z¯) between the
unknown spectrum X and the average spectrum Z

IN TANDEM WITH TEST_SET FILE, THIS FUNCTION WILL THEN PREDICT THE SPOKEN DIGITS THAT HAVE BEEN
STORED IN THE NPY FILE FROM TEST SET.

IN ORDER TO DO THIS, THIS FUNCTION COMPARES AGAINST THE AVERAGE SPECTRIUM OF THE TRAINING SET.
"""
import numpy as np
import time

from dft import dft
from recordsound import recordsound

def print_matrix(A, nr_decimals = 2):

    # Determine the number of digits in the largest number in the matrix and use
    # it to specify the number format

    nr_digits = np.maximum(np.floor(np.log10(np.amax(np.abs(A)))),0) + 1
    nr_digits = nr_digits + nr_decimals + 3
    nr_digits = "{0:1.0f}".format(nr_digits)
    number_format = "{0: " + nr_digits + "." + str(nr_decimals) + "f}"
    
    # Determine matrix size
    n = len(A)
    m = len(A[0])

    # Sweep through rows
    for l in range(m):
        value = " "

        # Sweep through columns
        for k in range(n):

            # ccncatenate entries to create row printout
            value = value + " " + number_format.format(A[k,l])

        # Print row
        print( value )

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency

    # loads test set
    test_set = np.load("test_set.npy")

    # loads (DFTs of) training set
    training_set_DFTs = np.abs(np.load("spoken_digits_DFTs.npy"))
    # Average spectra
    num_digits = len(training_set_DFTs)
    _, N = training_set_DFTs[0].shape
    average_spectra = np.zeros((num_digits, N), dtype=np.complex_)

    for i in range(num_digits):
        average_spectra[i, :] = np.mean(training_set_DFTs[i], axis=0) 

    num_recs, N = test_set.shape
    predicted_labels = np.zeros(num_recs)

    # Computes (normalized) DFTs of the test set
    DFTs_aux = np.zeros((num_recs, N), dtype=np.complex_)
    DFTs_c_aux = np.zeros((num_recs, N), dtype=np.complex_)
    
    for i in range(num_recs):
        rec_i = test_set[i, :]
        # We can use the norm of the ith signal to normalize its DFT
        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i
        DFT_rec_i = dft(rec_i, fs)
        [_, X, _, X_c] = DFT_rec_i.solve3()
        DFTs_aux[i, :] = X 
        DFTs_c_aux[i, :] = X_c

        # Inner products
        inner_prods = np.zeros(num_digits) 
        for j in range(num_digits):
            inner_prods[j] = np.inner(np.abs(X), np.abs(average_spectra[j, :]))
        predicted_labels[i] = np.argmax(inner_prods) + 1
    
    print("Average spectrum comparison --- predicted labels: \n")
    print_matrix(predicted_labels[:, None], nr_decimals=0)
    
    # Storing DFTs
    np.save("test_set_DFTs.npy", DFTs_aux)
    np.save("test_set_DFTs_c.npy", DFTs_c_aux)

    # Storing predicted labels
    np.save("predicted_labels_avg.npy", predicted_labels)
