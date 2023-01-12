# author: Marijn Venderbosch
# january 2023

from arc import Strontium88, LevelPlot
atom = Strontium88()

# variables

n_from = 4
n_to = 6

l_from = 0
l_to = 2

# plot levels
calc = LevelPlot(atom)
level = calc.makeLevels(n_from, n_to, l_from, l_to, sList=[0, 1])
calc.drawLevels('cm')
calc.showPlot()

