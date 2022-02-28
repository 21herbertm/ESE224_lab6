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

def print_matrix(A, num_of_decimals = 2):

    # NEED NUMBER OF DIGITS IN LARGEST NUMBER IN THE MATRIX
    # NEED TO FIND THE NUMBER FORMAT


    # Compare two arrays and returns a new array containing the element-wise maxima. If one of the elements being compared
    # is a NaN, then that element is returned. If both elements are NaNs then the first is returned.
    # The latter distinction is important for complex NaNs,
    # which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
    num_of_digits = np.maximum(np.floor(np.log10(np.amax(np.abs(A)))),0) + 1
    num_of_digits = num_of_digits + num_of_decimals + 3
    num_of_digits = "{0:1.0f}".format(num_of_digits)
    number_format = "{0: " + num_of_digits + "." + str(num_of_decimals) + "f}"
    
    # SET MATRIX SIZE
    columns= len(A)
    rows = len(A[0])

    # LOOPS THROUGH EACH ROW OF THE MATRIX OF DIGITS
    for l in range(rows):
        value = " "

        # LOOPS THROUGH EACH OF THE COLUMNS OF THE DIGITS THE ENCOMPASS THE MATRIX
        for k in range(columns):

            # ALLOCATING THE FINAL VALUE BY ADDTING IN EACH ITEM FROM L AND K LOOPS
            value = value + " " + number_format.format(A[k,l])

        # PRINTS THE FINAL ROW VALUE
        print( value )

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency

    # REFERENCE TO TEST_SET.PY FILE
    test_set = np.load("test_set.npy")

    # INSERTS THE DFTS FROM THE TRAINING SET
    # Calculate the absolute value element-wise.
    training_set_DFTs = np.abs(np.load("spoken_digits_DFTs.npy"))
    num_digits = len(training_set_DFTs)
    _, N = training_set_DFTs[0].shape
    average_spectra = np.zeros((num_digits, N), dtype=np.complex_)

    # LOOP FOR THE AVERAGE SPECTRA
    for i in range(num_digits):
        average_spectra[i, :] = np.mean(training_set_DFTs[i], axis=0) 

    num_of_recordings, N = test_set.shape
    predicted_labels = np.zeros(num_of_recordings)

    # FINDS A NORMALIZED DFT OF THE TEST SET
    DFTs_aux = np.zeros((num_of_recordings, N), dtype=np.complex_)
    DFTs_c_aux = np.zeros((num_of_recordingss, N), dtype=np.complex_)


    # LOOP THROUGH EACH OF THE RECORDINGS IN ORDER TO ALLOCATE THE NORM OF THE ITH SIGNAL
    #NORMALIZING THE DFT
    for i in range(num_of_recordings):
        """
        this is assigning a vector to a slice of numpy 2D array (slice assignment). 
        Self-contained example:

>>> import numpy
>>> a = numpy.array([[0,0,0],[1,1,1]])
>>> a[0,:] = [3,4,5]
>>> a
array([[3, 4, 5],
       [1, 1, 1]])
There is also slice assignment in base python, using only one dimension (a[:] = [1,2,3])
        
        """
        rec_i = test_set[i, :]

        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i
        DFT_rec_i = dft(rec_i, fs)
        [_, X, _, X_c] = DFT_rec_i.solve3()
        DFTs_aux[i, :] = X 
        DFTs_c_aux[i, :] = X_c

        # SETTING INNER PRODUCTS TO 0'S
        inner_prods = np.zeros(num_digits) 
        for j in range(num_digits):
            # ALLOCATING INNER PRODUCT
            inner_prods[j] = np.inner(np.abs(X), np.abs(average_spectra[j, :]))
        predicted_labels[i] = np.argmax(inner_prods) + 1

    # DISPLAYS THE AVERAGE SPECTRUM COMPARISON
    print("Average spectrum comparison --- predicted labels: \n")
    print_matrix(predicted_labels[:, None], num_of_decimals=0)
    
    # SAVES THE DFTS IN A NPY FILE
    np.save("test_set_DFTs.npy", DFTs_aux)
    np.save("test_set_DFTs_c.npy", DFTs_c_aux)

    # STORES PREDICTED VALUES IN NPY
    np.save("predicted_labels_avg.npy", predicted_labels)
