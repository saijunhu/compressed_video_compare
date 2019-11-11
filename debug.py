
from example import add_arrays_1d
from example import concat_arrays
import numpy as np

if __name__ == '__main__':
    var1= add_arrays_1d(np.arange(5),np.arange(5))
    print('var1',var1)
    var2 = concat_arrays(np.arange(5),np.arange(5))
    var2 = np.vstack(var2)
    print(var2.shape)
    # run()
