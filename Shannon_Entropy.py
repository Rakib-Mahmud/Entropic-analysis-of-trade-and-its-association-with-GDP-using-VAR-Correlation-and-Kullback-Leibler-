# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:00:17 2023

@author: Rakib Mahmud
"""
#Import All the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use('Solarize_Light2')
#Import Dataset
dataset = []
Pij = []

for x in range(1995,2018):
    dataset.append(pd.read_csv('allcom\Allcom{}.csv'.format(x),header=None))

#Country Codes of the used countries
countries = [156,276,392,842,251] #China,Germany,Japan,USA,France
#Filter only export values of the selected countries
for data in dataset:
    Pij.append(data[data[0].isin(countries) & (data[1]==2)])

#Calculate Pij first
for data in Pij:
    for country in countries:
        data.loc[data[0]==country,3] = data[data[0]==country][3]/np.sum(data[data[0]==country][3])
    
#Calculate entropies from Pij
entropies = []
for country in countries:
    entr = []
    for data in Pij:
        if data[data[0]==country][3].empty:
            entr.append(0)
        else:    
            pA = data[data[0]==country][3]
            Shannon = -np.sum(pA*np.log2(pA))
            entr.append(Shannon)
    entropies.append(entr)
    
# for i in range(0,5):
#     entropies[i] = entropies[i]/np.sum(entropies[i])

#Plot year-wise Curves of entropies for each country
years = [x for x in range(1995,2018)]

ymin = np.min([np.min(entropies[0]),np.min(entropies[1]),np.min(entropies[2]),np.min(entropies[3]),np.min(entropies[4])])
ymax = np.max([np.max(entropies[0]),np.max(entropies[1]),np.max(entropies[2]),np.max(entropies[3]),np.max(entropies[4])])

plt.figure(figsize=(9, 6))
plt.ylim(ymin-0.03,ymax+0.03)
plt.title('Evolution of Entropy (All Commodities)')
plt.xlabel('Year')
plt.ylabel('Entropy')
plt.xticks(years,rotation='45')

plt.plot(years,entropies[0],marker='s',markersize=5,linewidth=1,label='China',color='b')
plt.plot(years,entropies[1],marker='o',markersize=5,linewidth=1,label='Germany',color='g')
plt.plot(years,entropies[2],marker='v',markersize=5,linewidth=1,label='Japan',color='r')
plt.plot(years,entropies[3],marker='*',markersize=5,linewidth=1,label='USA',color='c')
plt.plot(years,entropies[4],marker='x',markersize=5,linewidth=1,label='France',color='m')
plt.legend(loc='best',ncol=3, fancybox=True, shadow=True)
plt.show()