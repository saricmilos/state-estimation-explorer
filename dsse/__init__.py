"""v2 of the thesis "Advanced State Estimation in Distribution Networks Using GNN and PINNs".

v1, the thesis as submitted, lives in its own repository and is only ever read: see dsse.v1.
"""

import sys

# Importing v1's modules from their folder would otherwise leave __pycache__ folders in the thesis
# repository, which must stay exactly as it was handed in.
sys.dont_write_bytecode = True
