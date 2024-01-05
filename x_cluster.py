# -*- coding: utf-8 -*-
"""
A class for interpreting XRF data using an unsupervised clustering approach. 
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class X_cluster():
    
    def __init__(self, folderpath = None):
        self.folderpath = folderpath
        self.text_files = None
    
    def set_folderpath(self, folderpath):
        #This method is only really necessary if the folderpath becomes a private or protected variable
        self.folderpath = folderpath
    
    def _identify_files(self):
        pass
    
    def _flatten_sheet(self):
        pass

    def _unflatten_sheet(self):
        pass
    
    def cluster(self):
        pass
        
    def _display_cluster(self):
        pass
    
    def display(self):
        pass