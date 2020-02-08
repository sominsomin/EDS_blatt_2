# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:51:31 2020

@author: Simon
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:16:57 2020

@author: Simon
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#Koeffizienten der Uebertragsfunktion 1
a1 = np.array([1])
b1 = np.array([0.5,-1,0.5])

#Uebertragsfunktion 2
a2 = np.array([1,-0.7,0.3])
b2 = np.array([0.5,-1,0.5])

# Samplingfrequenz in Hz:
Fs = 48000
# Frequenzvektor zum Plotten in Hz
f = np.arange(1,Fs)


#%%Frequenz-Antwort System 1
w, h = signal.freqz(b1, a1, worN=Fs)
#remove first sample which is zero
h = h[1:]

#Gruppenlaufzeit
wGd, gd = signal.group_delay((b1, a1), Fs)
#remove first sample which is zero
gd = gd[1:]

#phase delay
pd = np.angle(h)/(2*np.pi*f)

# Plotten
fig = plt.figure(figsize=(7.5,10))
plt.subplot(4,1,1)
# Betragsfrequenzgang
plt.semilogx(f, 20 * np.log10(abs(h)), 'b')
plt.title('Betragsfrequenzgang')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequenz [Hz]')
plt.grid(True)
# Phasenfrequenzgang
plt.subplot(4,1,2)
plt.semilogx(f, np.rad2deg(np.angle(h)), 'g')
plt.title('Phasenfrequenzgang')
plt.ylabel('Phase [Grad]')
plt.xlabel('Frequenz [Hz]')
plt.grid(True)
# Gruppenlaufzeit
plt.subplot(4,1,3)
plt.title('Gruppenlaufzeit')
plt.plot(f, gd)
plt.ylim([0,gd.max()+0.1])
plt.ylabel('Gruppenverzoegerung [samples]')
plt.xlabel('Frequenz [Hz]')
# Phasenlaufzeit
plt.subplot(4,1,4)
plt.title('Phasenlaufzeit')
plt.plot(f, pd)
plt.ylim([0, pd.max()])
plt.ylabel('Phaselaufzeit [samples]')
plt.xlabel('Frequenz [Hz]')

plt.tight_layout()

plt.savefig("aufgabe_2_frequ_response_system_1.png", dpi=150)
plt.show()



#%%Frequenz-Antwort System 2
w, h = signal.freqz(b2, a2, worN=Fs)
#remove first sample which is zero
h = h[1:]

#Gruppenlaufzeit
wGd, gd = signal.group_delay((b2, a2), Fs)
#remove first sample which is zero
gd = gd[1:]

#phase delay
pd = np.angle(h)/(2*np.pi*f)

# Plotten
fig = plt.figure(figsize=(7.5,10))
plt.subplot(4,1,1)
# Betragsfrequenzgang
plt.semilogx(f, 20 * np.log10(abs(h)), 'b')
plt.title('Betragsfrequenzgang')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequenz [Hz]')
plt.grid(True)
# Phasenfrequenzgang
plt.subplot(4,1,2)
plt.semilogx(f, np.rad2deg(np.angle(h)), 'g')
plt.title('Phasenfrequenzgang')
plt.ylabel('Phase [Grad]')
plt.xlabel('Frequenz [Hz]')
plt.grid(True)
# Gruppenlaufzeit
plt.subplot(4,1,3)
plt.title('Gruppenlaufzeit')
plt.plot(f, gd)
plt.ylim([0,gd.max()+0.1])
plt.ylabel('Gruppenverzoegerung [samples]')
plt.xlabel('Frequenz [Hz]')
# Phasenlaufzeit
plt.subplot(4,1,4)
plt.title('Phasenlaufzeit')
plt.plot(f, pd)
plt.ylim([0, pd.max()])
plt.ylabel('Phaselaufzeit [samples]')
plt.xlabel('Frequenz [Hz]')

plt.tight_layout()

plt.savefig("aufgabe_2_frequ_response_system_2.png", dpi=150)
plt.show()