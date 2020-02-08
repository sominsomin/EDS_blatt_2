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

def signalgenerator(wellenform, amplitude, abtastrate, grundperiode, signallaenge):
    """
    Eingabeparameter:
        wellenform      akzeptiert "sinus", "saegezahn", "dreieck", "rechteck"
        amplitude
        abtastrate      in Samples pro Sekunde
        grundperiode    in Samples
        signallaenge    in Sekunden
    """
    import numpy as np
    import scipy.signal
    import math
    
    if wellenform is "sinus":
        y = np.array([amplitude*np.sin(2*math.pi/grundperiode*i)
            for i in range(int(signallaenge*abtastrate))])
    if wellenform is "saegezahn":
        y = np.array([amplitude*scipy.signal.sawtooth(2*math.pi/grundperiode*i)
            for i in range(int(signallaenge*abtastrate))])
    if wellenform is "dreieck":
        y = np.array([amplitude*scipy.signal.sawtooth(2*math.pi/grundperiode*i, width=0.5)
            for i in range(int(signallaenge*abtastrate))])
    if wellenform is "rechteck":
        y = np.array([amplitude*scipy.signal.square(2*math.pi/grundperiode*i)
            for i in range(int(signallaenge*abtastrate))])  
    return y

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


#%%plot Musik
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

#%%plot Sprache
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

#%%sinus

amplitude = 1
sr = 48000
grundperiode = 10
#signallaenge in sekunden fuer 4 perioden
signallaenge = (4*grundperiode)/sr

signale_zeit = np.linspace(0,signallaenge, num=signallaenge*sr)

sinus = signalgenerator("sinus", amplitude, sr, grundperiode, signallaenge)
sinus_gefiltert_system1 = scipy.signal.lfilter(b1,a1,sinus)
sinus_gefiltert_system2 = scipy.signal.lfilter(b1,a1,sinus)

plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(signale_zeit, sinus)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.title('sinus mit Grund-Periode {}'.format(grundperiode))

plt.subplot(3,1,2)
plt.plot(signale_zeit, sinus_gefiltert_system1)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('sinus gefiltert mit System 1')

plt.subplot(3,1,3)
plt.plot(signale_zeit, sinus_gefiltert_system2)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('sinus gefiltert mit System 2')
plt.tight_layout()
plt.savefig('aufgabe_2_sinus_filtered.png')

#%%rechteck

rechteck = signalgenerator("rechteck", amplitude, sr, grundperiode, signallaenge)
rechteck_gefiltert_system1 = scipy.signal.lfilter(b1,a1,rechteck)
rechteck_gefiltert_system2 = scipy.signal.lfilter(b1,a1,rechteck)

plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(signale_zeit, rechteck)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.title('rechteck mit Grund-Periode {}'.format(grundperiode))

plt.subplot(3,1,2)
plt.plot(signale_zeit, rechteck_gefiltert_system1)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('rechteck gefiltert mit System 1')

plt.subplot(3,1,3)
plt.plot(signale_zeit, rechteck_gefiltert_system2)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('rechteck gefiltert mit System 2')
plt.tight_layout()
plt.savefig('aufgabe_2_rechteck_filtered.png')

#%%saegezahn

saegezahn = signalgenerator("saegezahn", amplitude, sr, grundperiode, signallaenge)
saegezahn_gefiltert_system1 = scipy.signal.lfilter(b1,a1,saegezahn)
saegezahn_gefiltert_system2 = scipy.signal.lfilter(b1,a1,saegezahn)

plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(signale_zeit, saegezahn)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.title('saegezahn mit Grund-Periode {}'.format(grundperiode))

plt.subplot(3,1,2)
plt.plot(signale_zeit, saegezahn_gefiltert_system1)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('saegezahn gefiltert mit System 1')

plt.subplot(3,1,3)
plt.plot(signale_zeit, saegezahn_gefiltert_system2)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('saegezahn gefiltert mit System 2')
plt.tight_layout()
plt.savefig('aufgabe_2_saegezahn_filtered.png')

#%%dreick

dreieck = signalgenerator("dreieck", amplitude, sr, grundperiode, signallaenge)
dreieck_gefiltert_system1 = scipy.signal.lfilter(b1,a1,dreieck)
dreieck_gefiltert_system2 = scipy.signal.lfilter(b1,a1,dreieck)

plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(signale_zeit, dreieck)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.title('dreieck mit Grund-Periode {}'.format(grundperiode))

plt.subplot(3,1,2)
plt.plot(signale_zeit, dreieck_gefiltert_system1)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('dreieck gefiltert mit System 1')

plt.subplot(3,1,3)
plt.plot(signale_zeit, dreieck_gefiltert_system2)
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.ylim([-1,1])
plt.title('dreieck gefiltert mit System 2')
plt.tight_layout()
plt.savefig('aufgabe_2_dreieck_filtered.png')