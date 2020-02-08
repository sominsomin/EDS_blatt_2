#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EDS Aufgabenbaltt 1
# Aufgabe 1

# Funktion definieren
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
            for i in range(signallaenge*abtastrate)])
    if wellenform is "saegezahn":
        y = np.array([amplitude*scipy.signal.sawtooth(2*math.pi/grundperiode*i)
            for i in range(signallaenge*abtastrate)])
    if wellenform is "dreieck":
        y = np.array([amplitude*scipy.signal.sawtooth(2*math.pi/grundperiode*i, width=0.5)
            for i in range(signallaenge*abtastrate)])
    if wellenform is "rechteck":
        y = np.array([amplitude*scipy.signal.square(2*math.pi/grundperiode*i)
            for i in range(signallaenge*abtastrate)])  
    return y


# Erzeugen der Signale
test_sinus      = signalgenerator("sinus", 1, 16000, 40, 2)
test_saegezahn  = signalgenerator("saegezahn", 1, 16000, 40, 2)
test_dreieck    = signalgenerator("dreieck", 1, 16000, 40, 2)
test_rechteck   = signalgenerator("rechteck", 1, 16000, 40, 2)


# Plotten der Signale zum Testen
# Um etwas zu erkennen nur Plotten der ersten 3 Perioden

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

plt.subplot(4,1,1)
plt.plot(test_sinus[:120])
plt.title("Sinus")
plt.xlabel("Zeit in Samples")
plt.ylabel("Amplitude")

plt.subplot(4,1,2)
plt.plot(test_saegezahn[:120])
plt.title("Sägezahn")
plt.xlabel("Zeit in Samples")
plt.ylabel("Amplitude")

plt.subplot(4,1,3)
plt.plot(test_dreieck[:120])
plt.title("Dreieck")
plt.xlabel("Zeit in Samples")
plt.ylabel("Amplitude")

plt.subplot(4,1,4)
plt.plot(test_rechteck[:120])
plt.title("Rechteck")
plt.xlabel("Zeit in Samples")
plt.ylabel("Amplitude")

plt.tight_layout()


# Abspielen zum Test
import sounddevice as sd
fs = 16000

sd.play(test_sinus, fs)
sd.wait()
sd.play(test_saegezahn, fs)
sd.wait()
sd.play(test_dreieck, fs)
sd.wait()
sd.play(test_rechteck, fs)
sd.wait()
