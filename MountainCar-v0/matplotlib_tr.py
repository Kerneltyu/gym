import math
import numpy as np
from matplotlib import pyplot

pi = math.pi

x = np.linspace(0, 2*pi, 100)
sin_y = np.sin(x)
cos_y = np.cos(x)

pyplot.plot(x, sin_y)
pyplot.plot(x, cos_y)

pyplot.title('Sin and Cos Graph')

pyplot.xlabel('X-Axis')
pyplot.ylabel('Y-Axis')

pyplot.legend()
pyplot.show()
