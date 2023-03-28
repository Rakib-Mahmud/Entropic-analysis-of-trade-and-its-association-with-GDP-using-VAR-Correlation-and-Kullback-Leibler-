# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 03:03:12 2023

@author: Rakib Mahmud
"""
#Import All the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr
from scipy.stats import entropy
from scipy.special import rel_entr
import seaborn as sn
plt.style.use('Solarize_Light2')

#Load Dataset
load = []

#Load Trade Values and Trade partners
for x in range(1995,2018):
    load.append(pd.read_csv('allcom\Allcom{}.csv'.format(x),header=None))
#Load GDPs of countries and drop rows having a nan value
gdp = pd.read_csv('GDP.csv')
gdp.dropna(inplace=True)
gdp.reset_index(inplace=True,drop=True)
#Load country code to country abbrev. map file
country_code = pd.read_excel('Comtrade Country Code and ISO list.xlsx')



#Country Codes of the used countries
countries = [156,276,392,842,251] #China,Germany,Japan,USA,France


#Add country codes to gdp file and sort gdps based on code
mapper = {}
rows,cols = country_code.shape
for i in range(0,rows):
    if country_code['ISO3-digit Alpha'][i] not in mapper.keys():
        mapper[country_code['ISO3-digit Alpha'][i]] = country_code['Country Code'][i]
        
r,c = gdp.shape
c_codes = []
for i in range(0,r):
    if gdp['Code'][i] not in mapper.keys():
        gdp.drop(labels=i,axis=0,inplace=True)
    else:
        c_codes.append(mapper[gdp['Code'][i]])
gdp.reset_index(inplace=True,drop=True)
gdp['Country Code'] = c_codes
gdp.sort_values(by=['Country Code'],inplace=True)
        

#Filter only export values of the selected countries
#Find out countries with a valid code, having GDP value available and delete the nan codes
valid_codes = country_code[(~pd.isna(country_code['ISO3-digit Alpha'])) & (country_code['ISO3-digit Alpha'].isin(gdp['Code']))]['Country Code'].values

#Extract common partner list of each counry for every year
common_partner = []
for country in countries:
    common_partner.append(list(set.intersection(*map(set, [load[x][(load[x][0]==country) & (load[x][1]==2) & (load[x][2].isin(valid_codes))][2] for x in range(0,23)]))))


#Rearrange the dataset so that only common partners retains for each year
df = []
for data in load:
    itr = 0
    df2 = pd.DataFrame()
    for country in countries:
        df2 = pd.concat([df2,data[(data[0]==country) & (data[2].isin(common_partner[itr])) & (data[1]==2)]]) 
        itr = itr+1
    df.append(df2)



#null_codes = country_code[pd.isna(country_code['ISO3-digit Alpha'])]['Country Code'].values


#Update trade values for each year based on GDP using VAR
Xij = []
#ss = []
for i in range(1,23):
    X = []
    for country in countries:
        X_prev = df[i-1][df[i-1][0]==country][3]
        partners = df[i][df[i][0]==country][2]
        names = country_code[country_code['Country Code'].isin(partners)]['ISO3-digit Alpha'].values
        g = gdp[gdp['Code'].isin(names)][str(1995+i)].values
        #ss.append(g)
        g_prev = gdp[gdp['Code'].isin(names)][str(1995+i-1)].values
        X.append((X_prev*g)/g_prev)
    Xij.append(np.concatenate([X[0].values,X[1].values,X[2].values,X[3].values,X[4].values]))



#Pearson Correlation
correlations = []
start = 0
for k in range(0,5):
    n_partner = len(common_partner[k])
    tmp_cor = []
    for i in range(0,22):
        original = df[i+1].iloc[start:start+n_partner,3].values
        update = Xij[i][start:start+n_partner]        
        corr, _ = pearsonr(original,update)
        tmp_cor.append(corr)
    correlations.append(tmp_cor)
    start = n_partner

    
#Relative Entropy
rel_entropy = []
# ele = []
start = 0
for k in range(0,5):
    n_partner = len(common_partner[k])
    tmp_entr = []
    for i in range(0,22):
        original = df[i+1].iloc[start:start+n_partner,3].values
        update = Xij[i][start:start+n_partner]        
        pk = original/np.sum(original) #Probabilities of original values
        qk = update/np.sum(update) #Probabilities of updated values
        tmp_entr.append(entropy(pk,qk,base=2))
    rel_entropy.append(tmp_entr)
    start = n_partner
    # ele.append(rel_entr(pk,qk)) #Elementwise Relative Entropy
        

#Plot Matrices
y_axis_labels = ['CHI','GER','JAP','USA','FRA'] #except CHILI
x_axis_labels = [y for y in range(1996,2018)]


#Plot Correlation Matrix
plt.figure()
ax = plt.axes()
sn.heatmap(correlations, annot=False,xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax,cbar_kws={'orientation': 'horizontal'},cmap="cubehelix")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
ax.set_title('Correlation (All Commodities)')
plt.show()



#Plot Relative Entropy Matrix
plt.figure()
ax = plt.axes()
sn.heatmap(rel_entropy, annot=False,xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax,cbar_kws={'orientation': 'horizontal'},cmap="cubehelix")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
ax.set_title('Relative Entropy (All Commodities)')
plt.show()
