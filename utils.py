import numpy as np
from cfg import *
from numpy.random import randn

def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/mu), mu))
    return (2*bit_r[:, 0]-1) + 1j*(2*bit_r[:, 1]-1)


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])


def removeCP(signal):
    return signal[CP:(CP+K)]


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))

    sigma2 = signal_power * 10**(-SNRdb/10)
    noise = np.sqrt(sigma2/2) * (randn(*convolved.shape) + 1j * randn(*convolved.shape))

    return convolved + noise


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def PS(bits):
    return bits.reshape((-1,))


def ofdm_simulate(codeword, channelResponse, SNRdb):
    OFDM_data = np.zeros(K, dtype=complex)
    pilotValue = Modulation(pilot_bits)
    OFDM_data[np.arange(K)] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFMD_TX = OFDM_withCP
    OFDM_RX = channel(OFMD_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    symbol = np.zeros(K, dtype=complex)
    codeword_qpsk = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qpsk
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_codeword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_codeword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)

    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))), 
                           np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))),  \
           abs(channelResponse)
