import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import ctypes
import platform ## Used in the c++ wrappers testing on which operating system we are working on
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()


#########################################################################################################
###### Wrapper functions to make Gaussian elimination.
###### Sped-up version targeting loss decoding, binary operations performed in Cpp

os_system = platform.system()
if os_system == 'Windows':
    # LTcpp_header = ctypes.cdll.LoadLibrary('./GaussElim.dll')
    LTcpp_header = ctypes.cdll.LoadLibrary('./GE_new_tb.dll')
    # try:
    #     LTcpp_header = ctypes.cdll.LoadLibrary('./libLossDec_win.dll')
    # except:
    #     LTcpp_header = ctypes.cdll.LoadLibrary(os.path.join(cwd, 'FusionLatticesAnalysis', 'libLossDec_win.dll'))
    print('Loaded C++ linear algebra functions for Windows OS')
elif os_system == 'Linux':
    try:
        LTcpp_header = ctypes.cdll.LoadLibrary('./libGaussElim.so')
    except:
        LTcpp_header = ctypes.cdll.LoadLibrary(os.path.join(cwd, 'FusionLatticesAnalysis', 'libLossDec.so'))
    print('Loaded C++ linear algebra functions for Linux OS')
else:
    raise ValueError('Os system not supported: only Windows or Linux')

LTcpp_header.LossDecoder_GaussElimin.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_int]
LTcpp_header.LossDecoder_GaussElimin_print.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_int]
LTcpp_header.LossDecoder_GaussElimin_trackqbts.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]
LTcpp_header.LossDecoder_GaussElimin_noordered_trackqbts.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]
if os_system == 'Windows':
    LTcpp_header.LossDecoder_GaussElimin_noordered_trackstabs.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]


def loss_decoding_gausselim_fast(m, num_lost_qbts, print = False):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    nr, nc = m.shape
    # start_t = default_timer()

    REF_m = m.copy()
    if print:
        LTcpp_header.LossDecoder_GaussElimin_print(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                             nr, nc, num_lost_qbts)
    else:
        LTcpp_header.LossDecoder_GaussElimin(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                             nr, nc, num_lost_qbts)
    return REF_m


def loss_decoding_gausselim_fast_trackqbts(m, qbt_syndr_mat, num_lost_qbts):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    nr, nc = m.shape
    # start_t = default_timer()
    if isinstance(m, (csr_matrix, csc_matrix)):
        # if copy_last:
        #     mdense = m.todense()
        #     REF_m = mdense.copy()
        # else:
        #     mcopy = m.copy()
        #     REF_m = mcopy.todense()
        # if check_m is not None:
        #     assert(np.allclose(REF_m, check_m))
        REF_m = m.toarray()
        # REF_m = m2.copy()
    else:
        REF_m = m.copy()
    REF_qbt_syndr_mat = qbt_syndr_mat.copy()
    LTcpp_header.LossDecoder_GaussElimin_trackqbts(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                   REF_qbt_syndr_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                   nr, nc, num_lost_qbts)
    return REF_m, REF_qbt_syndr_mat


def loss_decoding_gausselim_fast_noordering_trackstabs(m, lost_qbts, n_stabs):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    if lost_qbts.dtype != int:
        raise ValueError(f"The c++ function works only lost qubits with data type int, not {lost_qbts.dtype}")
    nr, nc = m.shape

    nlq = len(lost_qbts)
    lost_q = lost_qbts.copy()

    REF_m = m.copy()
    stab_arr = np.array(range(n_stabs), dtype=int)
    LTcpp_header.LossDecoder_GaussElimin_noordered_trackstabs(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                   stab_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                   lost_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                   nr, nc, nlq)
    return REF_m, stab_arr


def loss_decoding_gausselim_fast_noordering_trackqbts(m, qbt_syndr_mat, lost_qbts):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    if lost_qbts.dtype != int:
        raise ValueError(f"The c++ function works only lost qubits with data type int, not {lost_qbts.dtype}")
    nr, nc = m.shape

    nlq = len(lost_qbts)
    lost_q = lost_qbts.copy()

    REF_m = m.copy()
    REF_qbt_syndr_mat = qbt_syndr_mat.copy()
    LTcpp_header.LossDecoder_GaussElimin_noordered_trackqbts(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                   REF_qbt_syndr_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                   lost_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                   nr, nc, nlq)
    return REF_m, REF_qbt_syndr_mat

