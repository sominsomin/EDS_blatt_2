#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:49:27 2020
@author: die Gruppe


Blatt 2, Aufgabe 3

Die ermittelte Werte für Tief-, Hoch- und Bandpass sind bereits in Dateien gespeichert.

- Imports, Funktionen definieren
- Werte für Pol- und Nullstellen einpflegen und b und a berechnen lassen
- Filterungen vornehmen
- Normalisieren
- Abspielen
- Plotten der Impulsantworten, Betrags- und Phasengänge

"""

# Import der Bibliotheken
import scipy.signal
import scipy.io.wavfile
import sounddevice as sd
import numpy

# Import: Wieder Etablieren des Signalgenerators, Erzeugen Signale
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

test_sinus      = signalgenerator("sinus", 1, 16000, 40, 2)
test_saegezahn  = signalgenerator("saegezahn", 1, 16000, 40, 2)
test_dreieck    = signalgenerator("dreieck", 1, 16000, 40, 2)
test_rechteck   = signalgenerator("rechteck", 1, 16000, 40, 2)

# Einlesen der Audiosignale, Kanalreduktion beim Sprachsignal, Normalisieren
[fs_Audio, Sprache] = scipy.io.wavfile.read("Sprache.wav")
[fs_Audio, Musik]   = scipy.io.wavfile.read("Musik.wav")
Sprache = Sprache[:, 0]

# Funktion zum Normalisieren definieren
def normalize(v):
    '''
    "normalize" array to max value of 1
    '''
    maxValue = numpy.max(v)
    if maxValue == 0: 
       return v
    return v / maxValue

Sprache = normalize(Sprache)
Musik = normalize(Musik)

#

# Tiefpass
Polstelle_Tiefpass =    [0.7-0.1j, 0.7+0.1j]
Nullstelle_Tiefpass =   [-0.6+0.2j, -0.6-0.2j]

b_Tiefpass, a_Tiefpass = scipy.signal.zpk2tf(Nullstelle_Tiefpass, Polstelle_Tiefpass, 1)
print("b_Tiefpass =", b_Tiefpass)
print("a_Tiefpass =", a_Tiefpass)


# Hochpass
Polstelle_Hochpass =    [0.45-0.4j, 0.45+0.4j]
Nullstelle_Hochpass =   [0.95+0j, 0.95-0j]

b_Hochpass, a_Hochpass = scipy.signal.zpk2tf(Nullstelle_Hochpass, Polstelle_Hochpass, 1)
print("b_Hochpass =", b_Hochpass)
print("a_Hochpass =", a_Hochpass)


# Bandpass
Polstelle_Bandpass =    [0.001-0.001j, 0.001+0.001j]
Nullstelle_Bandpass =   [0.001-0.8j, 0.001+0.8j]

b_Bandpass, a_Bandpass = scipy.signal.zpk2tf(Nullstelle_Bandpass, Polstelle_Bandpass, 1)
print("b_Bandpass =", b_Bandpass)
print("a_Bandpass =", a_Bandpass)

#

# Filterungen
Sinus_Tiefpass      = scipy.signal.lfilter(b_Tiefpass, a_Tiefpass, test_sinus)
Saegezahn_Tiefpass  = scipy.signal.lfilter(b_Tiefpass, a_Tiefpass, test_saegezahn)
Dreieck_Tiefpass    = scipy.signal.lfilter(b_Tiefpass, a_Tiefpass, test_dreieck)
Rechteck_Tiefpass   = scipy.signal.lfilter(b_Tiefpass, a_Tiefpass, test_rechteck)
Sprache_Tiefpass    = scipy.signal.lfilter(b_Tiefpass, a_Tiefpass, Sprache)
Musik_Tiefpass      = scipy.signal.lfilter(b_Tiefpass, a_Tiefpass, Musik)

Sinus_Hochpass      = scipy.signal.lfilter(b_Hochpass, a_Hochpass, test_sinus)
Saegezahn_Hochpass  = scipy.signal.lfilter(b_Hochpass, a_Hochpass, test_saegezahn)
Dreieck_Hochpass    = scipy.signal.lfilter(b_Hochpass, a_Hochpass, test_dreieck)
Rechteck_Hochpass   = scipy.signal.lfilter(b_Hochpass, a_Hochpass, test_rechteck)
Sprache_Hochpass    = scipy.signal.lfilter(b_Hochpass, a_Hochpass, Sprache)
Musik_Hochpass      = scipy.signal.lfilter(b_Hochpass, a_Hochpass, Musik)

Sinus_Bandpass      = scipy.signal.lfilter(b_Bandpass, a_Bandpass, test_sinus)
Saegezahn_Bandpass  = scipy.signal.lfilter(b_Bandpass, a_Bandpass, test_saegezahn)
Dreieck_Bandpass    = scipy.signal.lfilter(b_Bandpass, a_Bandpass, test_dreieck)
Rechteck_Bandpass   = scipy.signal.lfilter(b_Bandpass, a_Bandpass, test_rechteck)
Sprache_Bandpass    = scipy.signal.lfilter(b_Bandpass, a_Bandpass, Sprache)
Musik_Bandpass      = scipy.signal.lfilter(b_Bandpass, a_Bandpass, Musik)

#

# Normalisieren
Sinus_Tiefpass      = normalize(Sinus_Tiefpass)
Saegezahn_Tiefpass  = normalize(Saegezahn_Tiefpass)
Dreieck_Tiefpass    = normalize(Dreieck_Tiefpass)
Rechteck_Tiefpass   = normalize(Rechteck_Tiefpass)
Sprache_Tiefpass    = normalize(Sprache_Tiefpass)
Musik_Tiefpass      = normalize(Musik_Tiefpass)

Sinus_Hochpass      = normalize(Sinus_Hochpass)
Saegezahn_Hochpass  = normalize(Saegezahn_Hochpass)
Dreieck_Hochpass    = normalize(Dreieck_Hochpass)
Rechteck_Hochpass   = normalize(Rechteck_Hochpass)
Sprache_Hochpass    = normalize(Sprache_Hochpass)
Musik_Hochpass      = normalize(Musik_Hochpass)

Sinus_Bandpass      = normalize(Sinus_Bandpass)
Saegezahn_Bandpass  = normalize(Saegezahn_Bandpass)
Dreieck_Bandpass    = normalize(Dreieck_Bandpass)
Rechteck_Bandpass   = normalize(Rechteck_Bandpass)
Sprache_Bandpass    = normalize(Sprache_Bandpass)
Musik_Bandpass      = normalize(Musik_Bandpass)

#

# Abspielen
fs       = 16000
fs_Audio = 44100

sd.play(test_sinus, fs)
sd.wait()
sd.play(Sinus_Tiefpass, fs)
sd.wait()
sd.play(Sinus_Hochpass, fs)
sd.wait()
sd.play(Sinus_Bandpass, fs)
sd.wait()

sd.play(test_saegezahn, fs)
sd.wait()
sd.play(Saegezahn_Tiefpass, fs)
sd.wait()
sd.play(Saegezahn_Hochpass, fs)
sd.wait()
sd.play(Saegezahn_Bandpass, fs)
sd.wait()

sd.play(test_dreieck, fs)
sd.wait()
sd.play(Dreieck_Tiefpass, fs)
sd.wait()
sd.play(Dreieck_Hochpass, fs)
sd.wait()
sd.play(Dreieck_Bandpass, fs)
sd.wait()

sd.play(test_rechteck, fs)
sd.wait()
sd.play(Rechteck_Tiefpass, fs)
sd.wait()
sd.play(Rechteck_Hochpass, fs)
sd.wait()
sd.play(Rechteck_Bandpass, fs)
sd.wait()

sd.play(Sprache, fs_Audio)
sd.wait()
sd.play(Sprache_Tiefpass, fs_Audio)
sd.wait()
sd.play(Sprache_Hochpass, fs_Audio)
sd.wait()
sd.play(Sprache_Bandpass, fs_Audio)
sd.wait()

sd.play(Musik, fs_Audio)
sd.wait()
sd.play(Musik_Tiefpass, fs_Audio)
sd.wait()
sd.play(Musik_Hochpass, fs_Audio)
sd.wait()
sd.play(Musik_Bandpass, fs_Audio)
sd.wait()