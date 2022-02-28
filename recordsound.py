import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

"""
GIVEN FOR LAB USE 
CITATION: https://ese224.seas.upenn.edu/ 
"""

class recordsound():
    """
    recordsound Record your voice for T time sampled at a frequency fs
    solve() generates a sampled signal of your voice and save it into a wav file
    """
    def __init__(self, T, fs):
        """
        :param T: the duration time
        :param fs: the sampling frequency
        """
        self.T = T
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the triangular pulse vector x
        :N: The sample duration
        """
        print('start recording \n')
        voicerecording = sd.rec(int(self.T * self.fs), self.fs, 1)
        sd.wait()  # Wait until recording is finished
        print('end recording \n')
        write('../../Desktop/myvoice1.wav', self.fs, voicerecording.astype(np.float32))  # Save as WAV file
        
        return voicerecording