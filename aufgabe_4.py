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

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k


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
a3 = [1, -1]
b3 = [1, -1]

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
plotAll(b3,a3,Fs, 'aufgabe_4_plot_3')
