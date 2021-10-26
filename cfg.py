import os
import numpy as np

K = 64     # 全部的子载波数
CP = K//4  # CP长度
P = 64     # 导频数

mu = 2   # QPSK调制
payloadBits_per_OFDM = K * mu
SNRdb = 20

pilot_bits = None
pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(pilot_file_name):
    print('load pilot txt')
    pilot_bits = np.loadtxt(pilot_file_name, delimiter=',')
else:
    pilot_bits = np.random.binomial(n=1, p=0.5, size=(K*mu,))
    np.savetxt(pilot_file_name, delimiter=',')

