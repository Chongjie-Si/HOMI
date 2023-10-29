import numpy as np
import arff


draft = arff.ArffDecoder().decode(open('./data/CAL500.arff', 'r'))
data = np.array(draft['data'], dtype=np.float32)
X = data[:, : 68]
Y = data[:, 68:]
print('CAL500')
