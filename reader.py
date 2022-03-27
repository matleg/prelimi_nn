import os
import numpy as np
import struct
import matplotlib.pyplot as plt

cwd = os.getcwd()
list_files = os.listdir(cwd)

with open(os.path.join(cwd, "data", "t10k-images-idx3-ubyte"), 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))  # ">":big endian, "I": unsigned int (4bytes by default)
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

print("magic", magic)
print("size", size)
print("nrows", nrows)
print("ncols", ncols)

plt.imshow(data[9,:,:], cmap='gray')
plt.show()

