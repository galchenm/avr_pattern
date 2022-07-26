#!/usr/bin/env python
# coding: utf8

"""
python parallel_av_pattern.py -i EuXFEL-S00008-r0096-c04.cxi EuXFEL-S00009-r0096-c00.cxi EuXFEL-S00013-r0096-c01.cxi EuXFEL-S00013-r0096-c02.cxi -p /entry_1/instrument_1/detector_1/detector_corrected/data
"""

import os
import sys
import h5py as h5
import numpy as np
import subprocess
import re
import argparse
from os.path import basename, splitext
import ctypes as ct

from multiprocessing import Pool, Lock, RawArray, Value

os.nice(0)

class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

def parse_cmdline_args():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)
    parser.add_argument('-i', nargs='+', type=str, help="List of cxi files")
    parser.add_argument('-p', type=str, help="hdf5 path for the cxi file data")

    return parser.parse_args()



def processing(i):
    global result_lock

    global I_xyz_arr
    global count_total_num_buf
    global shape
    global path_cxi

    print("FILENAME\n")
    
    file_cxi = i
    h5r = h5.File(file_cxi, 'r')
    IntData = h5r[path_cxi]
    shape_data = IntData.shape

    intensity = np.zeros(shape_data[1:])
    print('Number of patters for cxi file is {}\n'.format(shape_data[0]))
    num = shape_data[0]

    for i in range(num):
        intensity += IntData[i,]
    h5r.close()

    print('processing\n')
    with result_lock:
        I_xyz_arr += intensity
        count_total_num_buf.value += num



def init_worker(res_lck, I_xyz_buf):
    global result_lock

    global I_xyz_arr
    global shape
    print('INIT WORKER\n')
    I_xyz_arr = np.frombuffer(I_xyz_buf, dtype=np.double)
    I_xyz_arr = np.reshape(I_xyz_arr,shape)
    result_lock = res_lck
    
    

if __name__ == "__main__":
    
    args = parse_cmdline_args()

    list_of_cxi_files = args.i
    path_cxi = args.p
    
    num_events = subprocess.check_output(['/opt/hdf5/hdf5-1.10.6/bin/h5ls', str(list_of_cxi_files[0])+str(path_cxi)])
    num_events = num_events.strip().decode('utf-8').split('Dataset ')[1]
    num_events = re.sub(r'({|}|/Inf)', '', num_events).split(', ')

    if len(num_events) == 4:
        num, m, n, k = [int(i) for i in num_events]
        shape = (m, n, k)
        I_xyz_buf = RawArray(ct.c_double, m * n * k)
        print('Shape of av Int is ({}, {}, {})\n'.format(m, n, k))
    else:
        num, m, n = [int(i) for i in num_events]
        shape = (m, n)
        print('Shape of av Int is ({}, {})\n'.format(m, n))
        I_xyz_buf = RawArray(ct.c_double, m * n)
    
    count_total_num_buf = Value('i', 0, lock=True)
    result_lock = Lock()
    pool = Pool(31,
                initializer=init_worker,
                initargs=(result_lock, I_xyz_buf))

    pool.map(processing, list_of_cxi_files)
    I_xyz = np.frombuffer(I_xyz_buf, dtype=np.double)
    I_xyz = I_xyz.reshape(shape)
    

    print('Shape of last Int is {}\n'.format(I_xyz.shape))
    print('Total num is {}\n'.format(count_total_num_buf))
    
    I_xyz = I_xyz / count_total_num_buf.value
    
    f = h5.File('parallel-av-vds.h5', 'w')
    f.create_dataset('/data/data', data=np.array([I_xyz]))
    f.close()
