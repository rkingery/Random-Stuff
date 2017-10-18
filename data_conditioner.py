
# coding: utf-8

# In[44]:


import glob
import pandas as pd

path =r'/groups/LAARG/SHMMR/data/enc_stm/' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col=None, header=None, usecols=(0,2,6,7), names=('Date','AP MAC','Client MAC','Event'))
    list_.append(df)
dataframe = pd.concat(list_)
#dataframe = pd.read_csv("test.csv", header=None, usecols=(0,1,4,7), names=('Date','AP IP','Client IP','Event'))
dataframe.dropna()
dataframe = dataframe[dataframe['Event'] != 'Deauth to sta']
del dataframe['Event']
dataframe['AP MAC'] = pd.Categorical(dataframe['AP MAC'])
dataframe['Client MAC'] = pd.Categorical(dataframe['Client MAC'])
dataframe['APID'] = dataframe['AP MAC'].cat.codes
dataframe['ClientID'] = dataframe['Client MAC'].cat.codes
del dataframe['AP MAC']
del dataframe['Client MAC']
dataframe.to_csv(path_or_buf='/groups/LAARG/SHMMR/data/allDataGroupedBryan.csv')



