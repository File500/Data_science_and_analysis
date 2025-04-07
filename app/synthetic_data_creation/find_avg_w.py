# Ime 0.9, 0.76, 0.9, 0.7, 0.56, 0.2, 0.9
# Godina 0.8, 0.76, 0.5, 0.5, 0.7, 0.6, 1
# Cijena 1, 1, 0.46, 1, 1, 0.7, 1
# Kilometri 0.7, 0.9, 0.28, 0.45, 1, 0.3, 1
# Mjenjac 0.8, 1, 0.88, 0.5, 1, 0.3, 0.7
# Gorivo 0.6, 1, 1, 1, 1, 0.9, 0.7
# Vlasnik 0.2, 1, 1, 1, 0.3, 0.5, 0.7
# Prodavac 0.5, 1, 1, 1, 0.5, 0.2, 0.7

import numpy as np

tmp_arr = np.array([0.2, 1, 1, 1, 0.3, 0.5, 0.7])
print(tmp_arr.mean().round(decimals=4))