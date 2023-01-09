from arc import Rubidium, Strontium88, Wavefunction
import matplotlib.pyplot as plt

atom = Rubidium()
n = 10
l=3
j=3.5
mj=3.5

stateBasis = [[n, l, j, mj]]
stateCoef = [1] # pure 10 F_7/2 mj=7/2 state

wf = Wavefunction(atom, stateBasis, stateCoef)
wf.plot2D(plane="x-z", units="atomic") 
wf.plot2D(plane="x-y", units="atomic") 

plt.show()

atom2 = Strontium88()

n = 30
l=1
j=3
s=1
j=1

stateBasis2 = [[n, l, j, s]]
stateCoef2 = [1]

wf2 = Wavefunction(atom2, stateBasis2, stateCoef2, s=1)