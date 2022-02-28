"""
@author: Vinicius Lima
This function predicts spoken digits stored in "test_set.npy" by comparing their spectra
against each spectrum in the training set (stored in "spoken_digits_DFTs.npy").
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

            # concatenate entries to create row printout
            value = value + " " + number_format.format(A[k,l])

        # Print row
        print( value )

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency

    # loads test set
    test_set = np.load("test_set.npy")

    # loads (DFTs of) training set
    training_set_DFTs = np.load("spoken_digits_DFTs.npy")
    num_digits = len(training_set_DFTs)

    num_recs, N = test_set.shape
    predicted_labels = np.zeros(num_recs)

    # Computes (normalized) DFTs of the test set
    DFTs_aux = np.zeros((num_recs, N), dtype=np.complex_)
    DFTs_c_aux = np.zeros((num_recs, N), dtype=np.complex_)

    training_set_size, _ = training_set_DFTs[0].shape
    
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
        inner_prods = np.zeros((num_digits, training_set_size))
        for j in range(num_digits):
            for k in range(training_set_size):
                sample_dft = (training_set_DFTs[j])[k, :]  # from the training set
                inner_prods[j, k] = np.inner(np.abs(X), np.abs(sample_dft))
        max_position = np.unravel_index(np.argmax(inner_prods), inner_prods.shape)  
        predicted_labels[i] = max_position[0] + 1  # since inner_prods is a [digits \times samples] matrix
    
    print("Nearest neighbor comparison --- predicted labels: \n")
    print_matrix(predicted_labels[:, None], nr_decimals=0)
    
    # Storing DFTs
    np.save("test_set_DFTs.npy", DFTs_aux)
    np.save("test_set_DFTs_c.npy", DFTs_c_aux)

    # Storing predicted labels
    np.save("predicted_labels_nn.npy", predicted_labels)