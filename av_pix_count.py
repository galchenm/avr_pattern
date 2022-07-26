#!/usr/bin/env python3
# coding: utf8

"""
python3 ../av_pix_count.py -i EuXFEL-S00008-r0096-c04.cxi EuXFEL-S00009-r0096-c00.cxi EuXFEL-S00013-r0096-c01.cxi EuXFEL-S00013-r0096-c02.cxi -p /entry_1/instrument_1/detector_1/detector_corrected/data -m /entry_1/instrument_1/detector_1/detector_corrected/mask -out-prefix r0096-small

python3 ../av_pix_count.py -f files.lst -p /entry_1/instrument_1/detector_1/detector_corrected/data -m /entry_1/instrument_1/detector_1/detector_corrected/mask -out-prefix r0096-av-pix

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

import os
from collections import namedtuple


from multiprocessing import Pool, Lock, RawArray, Value
import time

MASK_BAD = int("0xfeff",16)
MASK_GOOD = int("0x0100",16)


os.nice(0)

class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

def parse_cmdline_args():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)
    parser.add_argument('-i', '--input', nargs='+', type=str, help="List of cxi files")
    parser.add_argument('-f', '--file', type=str, help="File with the list of cxi files")
    parser.add_argument('-p', type=str, help="hdf5 path for the cxi file data")
    parser.add_argument('-m', type=str, help="hdf5 path for the cxi file mask")
    parser.add_argument('-out-prefix', '--outPrefix', type=str, help="Prefix for the output h5 file")
    return parser.parse_args()



def processing(q):
    global result_lock

    global I_xyz_arr
    global I_xyz_num_arr
    global count_total_num_buf
    global shape
    global path_cxi
    global path_mask

    global count_dot

    file_cxi = q
    h5r = h5.File(file_cxi, 'r')
    IntData = h5r[path_cxi]
    MaskData = h5r[path_mask]

    shape_data = IntData.shape
    
    intensity = np.zeros(shape_data[1:])
    intensity_num = np.zeros(shape_data[1:])
    click_count = np.zeros(shape_data[1:])
    
    
    

    print('Number of patterns for {} cxi file is {}\n'.format(q, shape_data[0]))
    num = shape_data[0]

    '''
    /* Bad if it's missing any of the "good" bits */
    if ( (f & image->det->mask_good)
           != image->det->mask_good ) bad = 1;

    /* Bad if it has any of the "bad" bits. */
    if ( f & image->det->mask_bad ) bad = 1;
    

    print('Running...')
    for i in range(num):
        intensity_num += IntData[i,]
        bitwise_bad = np.bitwise_and(MaskData[i,], MASK_BAD)
        

        condition_good = bitwise_bad != MaskData[i,]
        condition_bad = bitwise_bad

        join_condition = np.logical_or(condition_good, condition_bad)

        intensity += np.where(join_condition == True, 0, IntData[i,]) #per-pixel intensity
        click_count += np.where(join_condition == True, 0, 1)
    '''

    '''
    /* Bad if it's missing any of the "good" bits */
    if ( (f & image->det->mask_good)
           != image->det->mask_good ) bad = 1;

    /* Bad if it has any of the "bad" bits. */
    if ( f & image->det->mask_bad ) bad = 1;
    '''

    print('Running...\n')
    for i in range(num):
        intensity_num += IntData[i,]
        bitwise_bad = np.bitwise_and(MaskData[i,], MASK_BAD)
        bitwise_good = np.bitwise_and(MaskData[i,], MASK_GOOD)

        condition_good = bitwise_good != MaskData[i,]
        condition_bad = bitwise_bad

        join_condition = np.logical_or(condition_good, condition_bad)

        intensity += np.where(join_condition == True, 0, IntData[i,]) #per-pixel intensity
        click_count += np.where(join_condition == True, 0, 1)
    
    print("Stop counting\n")
    h5r.close()

    print('processing\n')
    with result_lock:
        I_xyz_arr += intensity
        I_xyz_num_arr += intensity_num
        count_total_num_buf.value += num
        count_dot += click_count.astype(count_dot.dtype)
    print('stop processing for {}\n'.format(q))




def init_worker(res_lck, I_xyz_buf, I_xyz_num_buf, count_dot_buf):
    global result_lock
    global I_xyz_arr
    global I_xyz_num_arr
    global count_dot
    global shape

    print('INIT WORKER\n')
    I_xyz_arr = np.frombuffer(I_xyz_buf, dtype=np.double)
    I_xyz_arr = np.reshape(I_xyz_arr, shape)

    I_xyz_num_arr = np.frombuffer(I_xyz_num_buf, dtype=np.double)
    I_xyz_num_arr = np.reshape(I_xyz_num_buf, shape)

    count_dot = np.frombuffer(count_dot_buf, dtype=np.double)
    count_dot = np.reshape(count_dot, shape)

    result_lock = res_lck


if __name__ == "__main__":
    start_time = time.clock()
    args = parse_cmdline_args()

    list_of_cxi_files = []

    if args.input is not None:
        list_of_cxi_files = args.input
    else:
        try:
            with open(args.file, 'r') as f:
                for line in f:
                    list_of_cxi_files.append(line.strip())
        except:
            print("It's necessary to give a list of files or stream with them as input parameter.\n")


    path_cxi = args.p
    path_mask = args.m
    
    num_events = subprocess.check_output(['/opt/hdf5/hdf5-1.10.6/bin/h5ls', str(list_of_cxi_files[0])+str(path_cxi)])
    num_events = num_events.strip().decode('utf-8').split('Dataset ')[1]
    num_events = re.sub(r'({|}|/Inf)', '', num_events).split(', ')

    if len(num_events) == 4:
        num, m, n, k = [int(i) for i in num_events]
        shape = (m, n, k)
        print('Shape of av Int is ({}, {}, {})\n'.format(m, n, k))
        I_xyz_buf = RawArray(ct.c_double, m * n * k)
        I_xyz_num_buf = RawArray(ct.c_double, m * n * k)
        count_dot_buf = RawArray(ct.c_double, m * n * k)
    else:
        num, m, n = [int(i) for i in num_events]
        shape = (m, n)
        print('Shape of av Int is ({}, {})\n'.format(m, n))
        I_xyz_buf = RawArray(ct.c_double, m * n)
        I_xyz_num_buf = RawArray(ct.c_double, m * n)
        count_dot_buf = RawArray(ct.c_double, m * n)
    
    count_total_num_buf = Value('i', 0, lock=True)
    result_lock = Lock()

    if len(list_of_cxi_files) < 35:
        num_pool = len(list_of_cxi_files)
    else:
        num_pool = len(list_of_cxi_files)//2

    pool = Pool(num_pool,
                initializer=init_worker,
                initargs=(result_lock, I_xyz_buf, I_xyz_num_buf, count_dot_buf))

    pool.map(processing, list_of_cxi_files)
    I_xyz = np.frombuffer(I_xyz_buf, dtype=np.double)
    I_xyz = I_xyz.reshape(shape)
    I_xyz_num = np.frombuffer(I_xyz_num_buf, dtype=np.double)
    I_xyz_num = I_xyz_num.reshape(shape)

    count_dot = np.frombuffer(count_dot_buf, dtype=np.double)
    count_dot = count_dot.reshape(shape)
    

    print('Shape of last Int is {}\n'.format(I_xyz.shape))
    print('Total num is {}\n'.format(count_total_num_buf))
    
    I_xyz_av_num_patters = I_xyz_num / count_total_num_buf.value

    if args.outPrefix is None:
        f_name = 'parallel-av-num-patterns.h5'
        f2_name = 'parallel-av-count-dots.h5'
        f3_name = 'count-dots-data.h5'
    else:
        f_name = args.outPrefix + '-av-num-patterns.h5'
        f2_name = args.outPrefix + '-av-count-dots.h5'
        f3_name = args.outPrefix + '-count-dots-data.h5'
    
    f = h5.File(f_name, 'w')
    f.create_dataset('/data/data', data=np.array([I_xyz_av_num_patters]))
    f.close()

    index = (count_dot!= 0)
    I_xyz[index] = I_xyz[index] / count_dot[index]

    #I_xyz = np.where(count_dot != 0, I_xyz / count_dot, I_xyz)
    
    f2 = h5.File(f2_name, 'w')
    f2.create_dataset('/data/data', data=np.array([I_xyz]))
    f2.close()

    f3 = h5.File(f3_name, 'w')
    f3.create_dataset('/data/count', data=count_dot[index])
    f3.create_dataset('/data/num', data=count_total_num_buf.value)
    f3.close()
    
    print(time.clock() - start_time, "seconds")