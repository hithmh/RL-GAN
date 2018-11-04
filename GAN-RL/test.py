import numpy as np
s = np.zeros([42,42])
r = np.zeros([1])

transition = [s,[r]]
data = []
data.append(transition)
transition2 = transition
data.append(transition2)
print(data)