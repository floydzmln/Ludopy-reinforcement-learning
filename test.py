import numpy as np
import math
min_epsilon=0.001
decay_rate=0.995
episodes = math.log(min_epsilon / 1) / math.log(decay_rate)
print(episodes)
