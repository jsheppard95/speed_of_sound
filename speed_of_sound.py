#!/usr/bin/env python3
#
# speed_of_sound.py
# 
# Author: Jackson Sheppard
# 29Jul17 - Created, used wave module to read in audio file
# 30Jul17 - Opened wave file with scipy.io.wavfile: now can handle stereo
#         - Converted stereo to mono to take FFT
#         - Tested using constant 440 Hz tone - worked perfectly
#         - Tested using white noise, found v_sound = 321 m/s with first peak
# 31Jul17 - Wrote functions to find resonant frequencies from output data
#         - Found and displayed speed of sound: tested on tubes of two lengths
#         - Created nicer user interface, added axes titles to plots
#         - Added subprocess.call() function to record and create wave file
# 04Aug17 - Presentation:
#         - Change stereo to mono converter:
#            - Divides BEFORE adding when taking average (prevents overflow)
#            - Use list addition instead of for loop (Cuts run time in half)

# This program will take audio input as a .wav file of white noise played
# through an open-ended tube. It will measure the sound level vs. time of the
# clip and perform a fourier transform on the data to obtain an intensity vs.
# frequency spectrum. The peaks of maximum intensity correspond to resonant
# frequencies that are related to the speed of sound by:
# f_n = n*v/2L, where v is the speed of sound
# The program can thus determine the speed of sound from the resonant
# frequencies of the tube

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from matplotlib.mlab import psd
import subprocess
import sys
import time

# Use terminal command to either record or select wave file from working dir
filename = input('Enter filename, or type \'r\' to record new file: ')
if filename == 'r':
   filename = input('Enter name of new wave file: ')
   input('\nPress <Enter> to begin recording:\n')
   ret = subprocess.call(['arecord', '-f', 'dat', '-d', '20', '-D', 'plughw:1,0', filename])
   if ret != 0: # error
      print('Error Occured, check subprocess.call()')
      sys.exit()

LENGTH = .61595 # length of tube in meters
V_EXPECT = 343 # speed of sound in m/s

# scipy.io.wavefile.read automatically recognizes the stereo input
# and stores the data in an N by 2 array accordingly
# N (num_frames) = num_rows, 2 (num_channels) = num_cols
t0 = time.perf_counter()
print("Analzying Wave File...")
rate, data = scipy.io.wavfile.read(filename)

# Take transpose of data -> 2 by N array: [[1, 2, 3...], [4, 5, 6...]]
# This is easier to plot: each channel is stored in its own array
data = data.transpose()

# Plot original sound level:
# Plot each channel on same axes separately
# Now Plot sound level vs. time
num_frames = len(data[0]) # both channels have same num_frames
print('Number of frames:', num_frames)
print('')
print("Converting from frames to seconds...")
signal_time = num_frames/rate
tvals = np.linspace(0, signal_time, num_frames)
print("Signal time:", signal_time, 'seconds')
print('')

# Convert signal from stereo to mono for simpler fourier transform:
# Standard method is to average left and right channels:
print("Converting from Stereo to Mono...")
data_mono = np.zeros(num_frames)     # Added 04Aug 
data_mono = .5*data[0] + .5*data[1]  # faster than for loop
print('')

# Plot Mono sound level vs. time:
print("Plotting sound level vs. time...")
f1, ax1 = plt.subplots()
ax1.plot(tvals, data_mono)
ax1.set_xlabel('time, (seconds)')
ax1.set_ylabel('Sound Level')
ax1.set_title('Sound Level vs. Time')
f1.savefig(filename.rstrip('.wav') + '_soundvtime.png')
f1.show() # mono sound vs. time
print('')

# Now compute  power spectrum distribution of mono sound:
print("Transforming data set...")
ky, kx = psd(data_mono, NFFT=num_frames, Fs=rate)

# Plot intensity vs. frequency (PSD) of audio signal
print("Plotting Power Spectrum Distribution...")
f2, ax2 = plt.subplots()
ax2.plot(kx, ky)
ax2.set_xlim(0,2000)
ax2.set_xlabel('frequency, (Hz)')
ax2.set_ylabel('Intensity')
ax2.set_title('Intensity vs. Frequency')
f2.savefig(filename.rstrip('.wav') + '_psd.png')
f2.show()

# Functions to find peaks in transformed data:
def find_closest(array, value):
   """find closest entry within 'array' to desired 'value'
   returns closest entry in 'array'
   """
   delta_array = np.abs(array - value)
   index = np.argmin(delta_array)
   freq = array[index]
   return freq

def find_peaks(datax, datay, peak_width, num_peaks, V, L):
   """
   Function finds peaks corresponding to resonant frequencies in psd graph
   Returns array containing frequencies of maximum amplitude
   input: datax = xvals, datay=yvals, peak_width=range to search in Hz,
   num_peaks=number of peaks, V=speed of sound, L=length of tube
   """
   exp_freqs = np.zeros(num_peaks) # expected resonant freqs
   f_peaks = np.zeros(num_peaks)   # experimental resonant freqs
   for i in np.arange(num_peaks):
      exp_freqs[i] = (i+1)*V/(2*L) # f_n=n*v/(2*L)
   index = 0 # to be used for f_peaks
   # Search range of points surrounding expected frequencies for max amplitude
   for freq in exp_freqs:
      f_central = find_closest(datax, freq)  # find closest experimental freq
      f_left_guess = f_central - peak_width/2
      f_left = find_closest(datax, f_left_guess)
      f_right_guess = f_central + peak_width/2
      f_right = find_closest(datax, f_right_guess)
      # find indices of left and right bounds
      # np.nonzero(array == value): returns list of tuples containing indeces
      # that correspond to value
      # datax = 1d, increases linearly -> returns list with tuple with one entry
      left_indx_tup = np.nonzero(datax == f_left)
      left_indx = left_indx_tup[0][0]
      right_indx_tup = np.nonzero(datax == f_right)
      right_indx = right_indx_tup[0][0]
      # find max amplitude between left and right bound
      max_amp = 0
      for i in np.arange(left_indx, right_indx+1):
         if datay[i] > max_amp:
            max_amp = datay[i]
            max_f = datax[i]
      f_peaks[index] = max_f
      index += 1
   return f_peaks

print('')

# Find Resonant frequencies from PSD
peaks = find_peaks(kx, ky, 200, 4, V_EXPECT, LENGTH)
print("Measured Resonant Frequencies, in Hz:")
print(peaks)

# Plot f_n vs. n: f_n = n*v/(2*L)
# Fit linear equation: slope of best fit = v/2L
nvals = np.arange(1, len(peaks)+1)
f3, ax3 = plt.subplots()
ax3.plot(nvals, peaks, 'bo', label='data')
# create fit:
fit = np.polyfit(nvals, peaks, 1)
# fit equation: y = a*x + b
# returns array: [a, b]
fit_xvals = np.linspace(0, len(peaks), 10)
fit_yvals = (fit[0]*fit_xvals) + fit[1]
ax3.plot(fit_xvals, fit_yvals, 'r-', label='fit')
ax3.set_xlim([0, 4.5])
ax3.set_ylim([0, 1200])
ax3.set_xlabel('Harmonic number, n')
ax3.set_ylabel('Harmonic Frequencies, f_n')
ax3.set_title('Harmonic Frequencies vs. Harmonic Number')
ax3.legend(loc=2) # places legend in top left
f3.savefig(filename.rstrip('.wav') + '_fvn.eps')
f3.show()

print("")
slope = fit[0]
print("Slope of best fit:", slope, "Hz")
v_sound = 2*LENGTH*slope
print("Measured speed of sound:", v_sound, "m/s")
rel_err = (np.abs(V_EXPECT - v_sound)/V_EXPECT)*100
print("Relative error from speed of sound in air (343 m/s):", end=' ')
print('%.2f' % rel_err, end='%')
print('')
print('Time taken: %.2f' % (time.perf_counter() - t0), 's')

input("\nPress <Enter> to exit...\n")
