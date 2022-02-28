"""
MELANIE HERBERT
ALINA HO

ESE 224
LAB 6 VOICE RECOGNITION

ANOTHER METHOD TO PREDICT SPOKEN DIGITS THAT ARE RECIEVED AND STORED IN TEST_SET.NPY
COMPARING THE SPECTRA FROM TEST_SET.NPY  TO EACH SPECTRUM INSIDE OF THE TRAINING SET.

PART 3: NEAREST NEIGHBHOR COMPARISON

"""
import numpy as np
import time

from dft import dft
from recordsound import recordsound


"""
Nearest neighbor comparison. Compute the inner product p(X,Yi)
between the unknown spectrum X and each of the spectra Yi associated
4
with the word “one.” Do the same for the inner product p(X, Zi) 
between the unknown spectrum X and each of the spectra Zi associated
with the word “two.” Assign the digit of the spectrum with the largest
inner product. Estimate your classification accuracy

"""

def print_matrix(A, nr_decimals = 2):

    # Determine the number of digits in the largest number in the matrix and use
    # it to specify the number format

    num_of_digits = np.maximum(np.floor(np.log10(np.amax(np.abs(A)))),0) + 1

    """
    Compare two arrays and returns a new array containing the element-wise maxima. 
    If one of the elements being compared is a NaN, then that element is returned. If both elements are NaNs then 
    the first is returned. The latter distinction is important for complex NaNs, which are defined as at least one 
    of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
    """
    num_of_digits = num_of_digits + nr_decimals + 3
    num_of_digits = "{0:1.0f}".format(num_of_digits)
    number_format = "{0: " + num_of_digits + "." + str(nr_decimals) + "f}"

    # SET MATRIX SIZE
    columns = len(A)
    rows = len(A[0])

    # LOOPS THROUGH EACH ROW OF THE MATRIX OF DIGITS
    for l in range(rows):
        value = " "

        # LOOPS THROUGH EACH OF THE COLUMNS OF THE DIGITS THE ENCOMPASS THE MATRIX
        for k in range(columns):
            # ALLOCATING THE FINAL VALUE BY ADDTING IN EACH ITEM FROM L AND K LOOPS
            value = value + " " + number_format.format(A[k, l])

        # PRINTS THE FINAL ROW VALUE
        print(value)

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency

    # REFERENCE TO TEST_SET.PY FILE
    # loads test set
    test_set = np.load("test_set.npy")

    # loads (DFTs of) training set
    # INSERTS THE DFTS FROM THE TRAINING SET
    # Calculate the absolute value element-wise.
    training_set_DFTs = np.load("spoken_digits_DFTs.npy")
    num_digits = len(training_set_DFTs)

    num_recs, N = test_set.shape
    predicted_labels = np.zeros(num_recs)

    # FINDS A NORMALIZED DFT OF THE TEST SET
    DFTs_aux = np.zeros((num_recs, N), dtype=np.complex_)
    DFTs_c_aux = np.zeros((num_recs, N), dtype=np.complex_)

    matrix_size, _ = training_set_DFTs[0].shape
    
    for i in range(num_recs):
        rec_i = test_set[i, :]
        # We can use the norm of the ith signal to normalize its DFT
        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i
        DFT_rec_i = dft(rec_i, fs)
        [_, X, _, X_c] = DFT_rec_i.solve3()
        DFTs_aux[i, :] = X 
        DFTs_c_aux[i, :] = X_c

        inner_product = np.zeros((num_digits, matrix_size))
        for j in range(num_digits):
            for k in range(matrix_size):
                sample_dft = (training_set_DFTs[j])[k, :]  # from the training set
                inner_product[j, k] = np.inner(np.abs(X), np.abs(sample_dft))
        max_position = np.unravel_index(np.argmax(inner_product), inner_product.shape)
        predicted_labels[i] = max_position[0] + 1  # since inner_prods is a [digits \times samples] matrix

    # DISPLAYS THE NEAREST NEIGHBHOR COMPARISON
    print("Nearest neighbor comparison --- predicted labels: \n")
    print_matrix(predicted_labels[:, None], num_of_decimals=0)

    # SAVES THE DFTS IN A NPY FILE
    np.save("test_set_DFTs.npy", DFTs_aux)
    np.save("test_set_DFTs_c.npy", DFTs_c_aux)

    # STORES PREDICTED VALUES IN NPY
    np.save("predicted_labels_avg.npy", predicted_labels)