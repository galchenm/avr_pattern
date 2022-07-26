#!/usr/bin/env python
# coding: utf8

"""
python robust_reading.py -i EuXFEL-S00008-r0096-c04.cxi EuXFEL-S00009-r0096-c00.cxi EuXFEL-S00013-r0096-c01.cxi EuXFEL-S00013-r0096-c02.cxi -p /entry_1/instrument_1/detector_1/detector_corrected/data
"""

import os
import sys
import h5py as h5
import numpy as np
import subprocess
import re
import argparse
from os.path import basename, splitext


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


if __name__ == "__main__":
    
    args = parse_cmdline_args()

    list_of_cxi_files = args.i
    path_cxi = args.p
    
    num_events = subprocess.check_output(['/opt/hdf5/hdf5-1.10.6/bin/h5ls', str(list_of_cxi_files[0])+str(path_cxi)])
    num_events = num_events.strip().decode('utf-8').split('Dataset ')[1]
    num_events = re.sub(r'({|}|/Inf)', '', num_events).split(', ')

    if len(num_events) == 4:
        nm, m, n, k = [int(i) for i in num_events]
        Sum_Int = np.zeros((m, n, k))
        print('Shape of av Int is ({}, {}, {})\n'.format(m, n, k))
    else:
        nm, m, n = [int(i) for i in num_events]
        Sum_Int = np.zeros((m, n))
        print('Shape of av Int is ({}, {})\n'.format(m, n))
    num = 0
    for ind in range(0, len(list_of_cxi_files)):
        file_cxi = list_of_cxi_files[ind]
        h5r = h5.File(file_cxi, 'r')
        IntData = h5r[path_cxi]
        shape_data = IntData.shape
        num_patterns = shape_data[0]
        print('Number of patters for cxi file is {}\n'.format(num_patterns))
        
        num += num_patterns
        for i in range(num_patterns):
            Sum_Int += IntData[i,]
        h5r.close()
    
    print('Total num is {}\n'.format(num))
    print('Shape of av Int is ({})\n'.format(Sum_Int.shape))
    Sum_Int = Sum_Int / num
    
    f = h5.File('result-av-patterns-robust.h5', 'w')
    f.create_dataset('/data/data', data=Sum_Int)
    f.close()
