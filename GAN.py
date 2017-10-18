import numpy
import pandas as pd
import math
import time
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Masking, TimeDistributed, Reshape, Input, Dense, LSTM, Flatten, Bidirectional
from keras.models import load_model

# adjustable variables
sizeOfTraining = 10
minSize = 5
trainingEpochs = 100
train = True

# counter variables
i = -1
m = 0

# fix random seed for reproducibility
numpy.random.seed(7)

#33449157 total rows

# load the dataset
dataframe = pd.read_csv('/groups/LAARG/SHMMR/data/allDataGroupedBryan.csv', usecols=[2,3], engine='python', header=0, skipfooter=33399157)

# get max and min AP values for normalization
dataframeMin = float(dataframe.loc[:,'APID'].min())
dataframeMax = float(dataframe.loc[:,'APID'].max())

# sort by DEVICEID
df = dataframe.sort_values(by=['ClientID'], ascending=1)

# define data generator
def discrimDataGenerator():
    global i
    global trainingSet
    m = 0
    while True:
        if (m >= 15):
            trainX = numpy.random.rand(1, sizeOfTraining)
            trainX = trainX.reshape((1, sizeOfTraining))
            trainX = pad_sequences(trainX, maxlen=sizeOfTraining, 
                                   dtype='float32', padding='post', 
                                   truncating='post', value=0.)
            trainX = numpy.reshape(trainX, (1, sizeOfTraining))
            trainY = numpy.zeros((1, 1))
        else:
            trainX = trainingSet[m,0:sizeOfTraining]
            trainX = trainX.reshape((1, sizeOfTraining))
            trainX = pad_sequences(trainX, maxlen=sizeOfTraining, 
                                   dtype='float32', padding='post', 
                                   truncating='post', value=0.)
            trainX = numpy.reshape(trainX, (1, sizeOfTraining))
            trainY = numpy.ones((1, 1))
        m = m + 1
        if (m == i*2):
            m = 0
        yield (trainX, trainY)
		
def discrimBatchGenerator():
    global i
    global trainingSet
    m = 0
    trainX = trainingSet[m,0:sizeOfTraining]
    trainX = numpy.reshape(trainX, (1, trainX.shape[0]))
    trainX = pad_sequences(trainX, maxlen=sizeOfTraining, 
                            dtype='float32', padding='post', 
                            truncating='post', value=0.)
    trainX = numpy.reshape(trainX, (1, sizeOfTraining))
    trainY = numpy.ones((1, 1))
    m = m + 1
    if (m > i):
        m = 0
    return trainX, trainY

def genDataGenerator():
	global sizeOfInput
	global sizeOfTraining
	global trainingSet
	global i
	m = 0
	while True:
		trainX = trainingSet[m,0:sizeOfInput]
		trainY = trainingSet[m,0:sizeOfInput]
		trainY = numpy.reshape(trainY, (1, trainY.shape[0]))
		trainX = numpy.reshape(trainX, (1, trainX.shape[0]))
		trainX = pad_sequences(trainX, maxlen=sizeOfTraining, 
							   dtype='float32', padding='post', 
							   truncating='post', value=0.)		
		trainY = pad_sequences(trainY, maxlen=sizeOfTraining, 
							   dtype='float32', padding='post', 
							   truncating='post', value=0.)	 
		trainY = numpy.reshape(trainY, (1, sizeOfTraining, 1))
		trainX = numpy.reshape(trainX, (1, sizeOfTraining, 1))
		m = m + 1
		if (m >= i):
			m = 0
		yield (trainX, trainY)
		
def ganBatchGenerator():
	global sizeOfInput
	global sizeOfTraining
	while True:
		trainX = numpy.random.rand(1,1) 
		trainY = numpy.zeros((1,1))
		trainY = numpy.reshape(trainY, (1, 1))
		trainX = numpy.reshape(trainX, (1, trainX.shape[0]))
		trainX = pad_sequences(trainX, maxlen=sizeOfTraining, 
							   dtype='float32', padding='post', 
							   truncating='post', value=0.)		
		trainX = numpy.reshape(trainX, (1, sizeOfTraining,1))
		return trainX, trainY
	
def generator():
	modelG = Sequential()
	modelG.add(Bidirectional(LSTM(sizeOfTraining, return_sequences=True), input_shape=(sizeOfTraining,1)))
	modelG.add(TimeDistributed(Dense(1)))
	modelG.compile(loss='mean_squared_error', optimizer='adam')	 
	return modelG

def discriminator():
	modelD = Sequential()
	modelD.add(Dense(sizeOfTraining, input_dim=sizeOfTraining,
					kernel_initializer='normal', activation='relu'))
	modelD.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	modelD.compile(loss='binary_crossentropy', optimizer='adam')
	return modelD

def GAN(g, d):
	modelGAN = Sequential()
	modelGAN.add(g)
	modelGAN.add(Flatten())
	d.trainable=False
	modelGAN.add(d)
	modelGAN.compile(loss='binary_crossentropy', optimizer='adam')
	return modelGAN

def GANdataPrep(sizeOfInput):
	global i  
	global trainingSet
	trainingSet = numpy.empty((1, sizeOfInput+1))
	# iterate on each unique DEVICEID to create a new dataset
	for device, df_device in df.groupby('ClientID'):
		# select only the AP column
		dataframe = df_device.loc[:,['APID']]
		# reset the row index number
		dataframe.reset_index(drop=1, inplace=True)
	
		dataframe = dataframe.astype('Float32')

		# normalize AP values
		for j in range(0,len(dataframe[1:])+1):
			dataframe['APID'][j] = (dataframe['APID'][j] - dataframeMin) / (dataframeMax - dataframeMin)

		# convert to numpy ndarray
		dataset = dataframe.values
	
		# rows of sequences not columns
		dataset = dataset.transpose()
		# produce a new dataset for training
		for windowStart in range(0, ((dataset.size)-sizeOfInput+1)):       
			if ((dataset.size-windowStart) >= (sizeOfInput + 1)):
				i = i + 1
				if (i == 0):
					trainingSet[0] = dataset[:,windowStart:(windowStart+sizeOfInput+1)]
				else:
					trainingSet = numpy.append(trainingSet, dataset[:,windowStart:(windowStart+sizeOfInput+1)], axis=0)

gen = generator()
#print(gen.summary())
discrim = discriminator()
#print(discrim.summary())
#gen.load_weights('gen_save.h5')
#discrim.load_weights('disc_save.h5')

GAN = GAN(gen, discrim)
#print(GAN.summary())
#GAN.load_weights('gan_save.h5')
disc_loss = []
gan_loss = []

if train:
    for sizeOfInput in range(minSize, sizeOfTraining+1):
        i=-1
        GANdataPrep(sizeOfInput)  
        gen.fit_generator(genDataGenerator(), steps_per_epoch=i, 
                      epochs=trainingEpochs, verbose=0)
    
    gen.save_weights('gen_save.h5')

    
    i=-1
    GANdataPrep(sizeOfTraining)  
    discrim.fit_generator(discrimDataGenerator(), steps_per_epoch=i, 
                     epochs=trainingEpochs, verbose=0)
    
    discrim.save_weights('disc_save.h5')            
    
    GANdataPrep(sizeOfTraining)

    for epoch in range(1, trainingEpochs):
        trainX, trainY = discrimBatchGenerator()
        disc_loss.append(discrim.train_on_batch(trainX, trainY))
        trainX, trainY = ganBatchGenerator()
        gan_loss.append(GAN.train_on_batch(trainX, trainY))
        #print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, gan_loss[-1], disc_loss[-1]))

    GAN.save_weights('gan_save.h5')    
print(disc_loss)
print(gan_loss)


