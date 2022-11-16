from arc import *

sr88 = Strontium88()

n_lower = int(40)
n_higher = int(100)
n_span = int(n_higher - n_lower)
n_array = np.linspace(n_lower, n_higher, n_span + 1)

lifetime_list = []

for n in n_array:
    lifetime = sr88.getStateLifetime(n, 0, 1, s=1)
    lifetime_list.append(lifetime)