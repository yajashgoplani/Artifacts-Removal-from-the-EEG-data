# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 18:24:20 2019

@author: HP
"""

import Lbeep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
d=Lbeep.read_cnt('mn.cnt')
s=d.get_sample_count()
fs=d.get_sample_frequency()
c=d.get_samples(1,s)
q=np.array(c)
q1=q.reshape(962335,64)
ch_names=['GND','FP1','FP2','FPz','AFz','AF3','AF7','AF4','AF8','Fz','F2','F4','F6','F8','F1','F3','F5','F7','FCz','FC2','FC4','FC6','FT8','FC1','FC3','FC5','FT7','Cz','C2','C4','C6','T8','C1','C3','C5','T7','CPz','CP2','CP4','CP6','TP8','CP1','CP3','CP5','TP7','Pz','P2','P4','P6','P8','P1','P3','P5','P7','PO3','PO5','PO7','POz','PO4','PO6','PO8','Oz','O1','O2']
info = mne.create_info(ch_names, 1024, ch_types=["eeg"] * 64)
raw = mne.io.RawArray(q1.T, info)
raw.set_montage("standard_1020")
raw_tmp = raw.copy()
raw_tmp.filter(1, None)
ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
ica.fit(raw_tmp)