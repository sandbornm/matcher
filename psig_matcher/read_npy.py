import glob
import numpy as np
from utils import load_np

npy_files = glob.glob('./data/*/*.npy')

print(f"found {len(npy_files)} npy files")

npy_data = []
for npy_file in npy_files:
    data = np.load(npy_file)
    print(npy_file)
    print(data.shape)
    npy_data.append(data)
"""
example output:
./data/FLG/FLG1_3.npy
(400, 2)

400 samples of 2 columns (frequency (hz), real impedance (ohms))
"""
read_npy()

frequency_hz = npy_data[0][:, 0]
impedance_ohms = npy_data[0][:, 1]
print(npy_data[0])

"""
The goal is to correctly classify instance level signatures to the correct part instance

Example:

Part type: BOX 
Part instance: BOX1 - instance 1 of part type BOX
Part instance signature: BOX1_1 - first measured signature of instance 1 of part type BOX

BOXA_X --> BOXA ie all measured signatures map to BOX instance A
BOXB_Y --> BOXB ie all measured signatures map to BOX instance B
"""