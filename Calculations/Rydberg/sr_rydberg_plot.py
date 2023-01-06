from arc import *

atom = Strontium88()
calc = LevelPlot(atom)

n_from = 4
n_to = 6

l_from = 0
l_to = 2

level = calc.makeLevels(n_from, n_to, l_from, l_to, sList=[0, 1])
calc.drawLevels('cm')
calc.showPlot()

