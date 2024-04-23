# -*- coding: utf-8 -*-
"""Data Loader"""

import pandas as pd
import h5py
import sys
import numpy as np

class DataLoader:
    """Data Loader class"""
    
    def __init__(self, data):

        self.data = data
       
    def load_data(self, data_config):
        """Loads dataset from path"""
        
        # Check training or testing data
        if(self.data == 'COMPAS'):
            df = h5py.File(data_config.path_COMPAS_data, 'r')

        elif(self.data == 'mock'):
            df = h5py.File(data_config.path_mock_data, 'r')

        samples = df['sf_samples'][()]
        lnl_values = df['lnl'][()]

        return samples, lnl_values

    
    