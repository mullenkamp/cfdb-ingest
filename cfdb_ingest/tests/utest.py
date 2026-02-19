#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:36:32 2026

@author: mike
"""
import cfdb_ingest
import cfdb
import pathlib
import xarray as xr
import numpy as np

######################################################
### Parameters

base_dir = pathlib.Path('/home/mike/data/wrf/tests/physics_tests/2023-02-10/d03_SMS-3DTKE')

files = ['wrfout_d03_2023-02-12_00:00:00.nc', 'wrfout_d03_2023-02-13_00:00:00.nc']


output_path = pathlib.Path('/home/mike/data/wrf/tests/physics_tests/2023-02-10/gabriele_3km.cfdb')





#####################################################
### Tests

wrf1 = cfdb_ingest.WrfIngest([base_dir.joinpath(file) for file in files])

wrf1.convert(output_path, target_levels=[20, 50, 100])

d1 = cfdb.open_dataset(output_path)

air_temp = d1['air_temperature']

air_temp1 = air_temp.loc['2023-02-12T00:00', 2, :, :]


x1 = xr.open_dataset(base_dir.joinpath(files[0]))

t2 = x1['T2']

if not np.allclose(t2[0, :, :].values.round(2), air_temp1.values.squeeze(), rtol=0.01):
    raise ValueError()


precip = d1['precipitation']
precip1 = precip.loc['2023-02-12T01:00', 0, :, :]

rainc = x1['RAINC']
rainnc = x1['RAINNC']

rainc1 = rainc[:2, :, :].values
rainnc1 = rainnc[:2, :, :].values

combo = (rainc1 + rainnc1)

precip1.values.round(2)


precipc = x1['PREC_ACC_C']
precipnc = x1['PREC_ACC_NC']
































































































