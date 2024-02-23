# import the CM_rand.py in the brazil directory
from CM_rand import config, CM
import numpy as np

args = config()
B1 = np.array([[1, 0, 0, 0],
               [1, 1, 0, 0]])
S1 = np.eye(2)
B2 = np.array([[1, 0, 0, 0],
               [1, 1, 0, 0],
               [1, 1, 1, 0],
               [1, 1, 1, 1]])
S2 = np.eye(4)

com_mech = CM(args, B1, S1, B2, S2)
B_com = com_mech.Bc
S_com = com_mech.Sc
