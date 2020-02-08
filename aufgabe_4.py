# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:57:02 2020

@author: Simon
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

#Koeffizienten der Uebertragsfunktion 1
#m > n
a1 = [1, 0, -1]
b1 = [1, -1]

#Koeffizienten der Uebertragsfunktion 2
#m < n
a2 = [1, -1]
b2 = [1, 0, -1]

#Koeffizienten der Uebertragsfunktion 3
#m = n
#a3 = [1, -1]
#b3 = [1, -1]

# Samplingfrequenz in Hz:
Fs = 48000
# Frequenzvektor zum Plotten in Hz
f = np.arange(1,Fs)

def plotAll(b,a, Fs, title) :
    #Frequenz-Antwort
    w, h = signal.freqz(b, a, worN=Fs)
    #remove first sample which is zero
    h = h[1:]
    
    #Gruppenlaufzeit
    wGd, gd = signal.group_delay((b, a), Fs)
    #remove first sample which is zero
    gd = gd[1:]
    
    #phase delay
    pd = np.angle(h)/(2*np.pi*f)
    
    # Plotten
    plt.figure(figsize=(7.5,10))
    plt.subplot(3,1,1)
    # Betragsfrequenzgang
    plt.semilogx(f, 20 * np.log10(abs(h)), 'b')
    plt.title('Betragsfrequenzgang')
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequenz [Hz]')
    plt.grid(True)
    # Phasenfrequenzgang
    plt.subplot(3,1,2)
    plt.semilogx(f, np.rad2deg(np.angle(h)), 'g')
    plt.title('Phasenfrequenzgang')
    plt.ylabel('Phase [Grad]')
    plt.xlabel('Frequenz [Hz]')
    plt.grid(True)
    # Gruppenlaufzeit
    plt.subplot(3,1,3)
    plt.title('Gruppenlaufzeit und Phasenlaufzeit')
    plt.plot(f, gd)
    plt.plot(f, pd)
    plt.ylim([0,np.array([pd.max(),gd.max()]).max()])
    plt.ylabel('Verzoegerung [samples]')
    plt.xlabel('Frequenz [Hz]')
    plt.legend(['Gruppenlaufzeit','Phasenlaufzeit'])
    
    plt.tight_layout()
    
    plt.savefig("{}.png".format(title), dpi=150)
    plt.show()


#%%
plotAll(b1,a1,Fs, 'aufgabe_4_plot_1')
plotAll(b2,a2,Fs, 'aufgabe_4_plot_2')
#plotAll(b3,a3,Fs, 'aufgabe_4_plot_3')
