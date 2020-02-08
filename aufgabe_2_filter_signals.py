# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:11:32 2020

@author: Simon
"""

import numpy as np
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd

def normalize(v):
    '''
    "normalize" array to max value of 1
    '''
    maxValue = np.max(v)
    if maxValue == 0: 
       return v
    return v / maxValue

def makeMono(x) :
    '''
    take stereo signal and return mono signal
    '''
    y = np.zeros(len(x))
    for i, sample in enumerate(x) :
        y[i] = 0.5*sample[0] + 0.5*sample[1]
    return y

#Koeffizienten der Uebertragsfunktion 1
a1 = np.array([1])
b1 = np.array([0.5,-1,0.5])

#Koeffizienten der Uebertragsfunktion 2
a2 = np.array([1,-0.7,0.3])
b2 = np.array([0.5,-1,0.5])


musik_sr, musik = scipy.io.wavfile.read('musik.wav')
sprache_sr, sprache = scipy.io.wavfile.read('sprache.wav')

musik = normalize(musik)
sprache = makeMono(sprache)
sprache = normalize(sprache)

musik_time = [1/musik_sr*n for n in range(len(musik))]

musik_filtered_system_1 = scipy.signal.lfilter(b1,a1,musik)
musik_filtered_system_2 = scipy.signal.lfilter(b2,a2,musik)

sprache_time = [1/sprache_sr*n for n in range(len(sprache))]

sprache_filtered_system_1 = scipy.signal.lfilter(b1,a1,sprache)
sprache_filtered_system_2 = scipy.signal.lfilter(b2,a2,sprache)

#play audio one after the other
#uncomment to play the audio
sd.play(musik_filtered_system_1, musik_sr)
sd.wait()
sd.play(musik_filtered_system_2, musik_sr)
sd.wait()
sd.play(sprache_filtered_system_1, sprache_sr)
sd.wait()
sd.play(sprache_filtered_system_2, sprache_sr)
sd.wait()


#plot Musik
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(musik_time, musik)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.title('musik.wav')

plt.subplot(3,1,2)
plt.plot(musik_time, musik_filtered_system_1)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('musik.wav gefiltert mit System 1')

plt.subplot(3,1,3)
plt.plot(musik_time, musik_filtered_system_2)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('musik.wav gefiltert mit System 2')
plt.tight_layout()
plt.savefig('musik_filtered.png')

#plot Sprache
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(sprache_time, sprache)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.title('sprache.wav')

plt.subplot(3,1,2)
plt.plot(sprache_time, sprache_filtered_system_1)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('sprache.wav gefiltert mit System 1')

plt.subplot(3,1,3)
plt.plot(sprache_time, sprache_filtered_system_2)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('sprache.wav gefiltert mit System 2')
plt.tight_layout()
plt.savefig('sprache_filtered.png')