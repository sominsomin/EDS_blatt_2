# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:15:05 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getBetragUebertragsfunktion(a,b,z) :
    '''
    berechne H(z) fuer ein gegebenen Wert fuer z
    '''
    zaehler = np.polyval(a,z)
    nenner = np.polyval(b,z)
    
    H_betrag = 20*np.log(np.absolute(nenner/zaehler))
    
    return H_betrag

#Koeffizienten der Uebertragsfunktion 1
a1 = np.array([1])
b1 = np.array([0.5,-1,0.5])

#Koeffizienten der Uebertragsfunktion 2
a2 = np.array([1,-0.7,0.3])
b2 = np.array([0.5,-1,0.5])

#die Aufloesung ist hoeher als in der Aufgabenstellung
x = np.linspace(-1.5,1.5, int(3/0.025)+100)
y = np.linspace(-1.5,1.5, int(3/0.025)+100)

xi, yi = np.meshgrid(x,y)

H_1_surface = np.array([[0 for yj in y] for xi in x])
H_2_surface = np.array([[0 for yj in y] for xi in x])
for i in range(len(xi)) :
    for j in range(len(yi)) :
        z = xi[i,j] + yi[i,j]*1j
        #print(z)
        H_1_surface[i,j] = getBetragUebertragsfunktion(a1,b1,z)
        H_2_surface[i,j] = getBetragUebertragsfunktion(a2,b2,z)

#Koordinate fuer den Kreis
theta = np.linspace(0, 2 * np.pi, 201)
x_circ = 1*np.cos(theta)
y_circ = 1*np.sin(theta)

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Realteil')
ax.set_ylabel('Imaginaerteil')
ax.set_zlabel('Betrag der Uebertragsfunktion |H(z)| [dB]')
plt.title('Betrag der Uebertragsfunktion |H(z)| in der komplexen Ebene, System 1')
ax.plot(x_circ,y_circ)
mycmap = plt.get_cmap('coolwarm')
surf1 = ax.plot_surface(xi, yi, H_1_surface, alpha=0.6,cmap=mycmap)
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5, label='|H(z)| [db]')
plt.savefig('surface_plot_1.pdf')

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Realteil')
ax.set_ylabel('Imaginaerteil')
ax.set_zlabel('Betrag der Uebertragsfunktion |H(z)| [dB]')
plt.title('Betrag der Uebertragsfunktion |H(z)| in der komplexen Ebene, System 2')
ax.plot(x_circ,y_circ)
mycmap = plt.get_cmap('coolwarm')
surf2 = ax.plot_surface(xi, yi, H_2_surface, alpha=0.6,cmap=mycmap)
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=5, label='|H(z)| [db]')
plt.savefig('surface_plot_2.pdf')

plt.show()

