# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:29:22 2019

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import Lbeep
d=Lbeep.read_cnt('mi.cnt')
s=d.get_sample_count()
fs=d.get_sample_frequency()
c=d.get_samples(1,s)
q=np.array(c)
q1=q.reshape(618483,64)
q2=pd.DataFrame(q1)
x=StandardScaler()
x.fit(q2)
q3=x.transform(q2)
q4=pd.DataFrame(q3)
pca=PCA()
pca.fit(q4)
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
labels=['PC'+str(x) for x in range(1,len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.show()
pca_df=pd.DataFrame([per_var],columns=labels)
plt.scatter(pca_df.PC1,pca_df.PC2)
for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
plt.show()

pca=PCA(n_components=2)
q5=pca.fit_transform(q4)
plt.scatter(q5[:,0],q5[:,1])
plt.show()