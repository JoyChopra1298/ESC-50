import utils
from clip import Clip
from model import Model
from keras.optimizers import SGD
import keras.metrics as metrics
from keras.callbacks import ModelCheckpoint,CSVLogger
import keras
import numpy as np
from keras.models import model_from_json
import os

####### Load dataset
clips = utils.load_dataset("./ESC-50-master/pickled")

def get_features(clip):
	return np.stack((clip.PElog_spectra[:,:,i],clip.PElog_delta[:,:,i]),axis=-1)

def train(save_path,fold=1):
	# Note save_path should be a directory

	####### Separate segments for each fold and create 2x60x41 input for the neural network.
	x = {}
	y = {}
	for fold in clips:
		fold_x = []
		fold_y = []    
		for clip in clips[fold]:
			for i in range(clip.PElog_spectra.shape[2]):
				fold_y.append(keras.utils.to_categorical(clip.target,num_classes=50))
				fold_x.append(get_features(clip))
		x[int(fold)] = np.array(fold_x)
		y[int(fold)] = np.array(fold_y)

	model = Model()
	model.fold = fold

	sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
	model.model.compile(optimizer=sgd,loss="categorical_crossentropy", metrics=['accuracy'])
	
	x_train =[]
	y_train =[]

	x_validate =[]
	y_validate =[]

	for _fold in x:
		if(_fold!=model.fold):
			x_train.extend(x[_fold])
			y_train.extend(y[_fold])
		else:
			x_validate.extend(x[_fold])
			y_validate.extend(y[_fold])
	
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_validate = np.array(x_validate)
	y_validate = np.array(y_validate)

	os.makedirs(save_path, exist_ok=True)
	print(x_train.shape,y_train.shape,x_validate.shape,y_validate.shape)
	csv_logger = CSVLogger(save_path+"log.csv", append=True)
	checkpointer = ModelCheckpoint(save_path+"weights_best.hdf5", monitor='val_acc',save_weights_only=True,save_best_only=True,verbose=1, period=1)
	
	model.model.fit( x=x_train, y=y_train, batch_size=200, epochs=300, verbose=1, callbacks=[csv_logger,checkpointer], validation_data=(x_validate,y_validate),initial_epoch=model.epochs)

def predict(load_path):
	model = Model()
	model.model.load_weights(load_path+"weights_best.hdf5")
	
	sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
	model.model.compile(optimizer=sgd,loss="categorical_crossentropy", metrics=['accuracy'])

	for fold in clips:
		print(fold)
		for clip in clips[fold]:
			print(clip)
			x = [] 
			x.append(get_features(clip))
			x = np.array(x)
			clip_prediction = model.model.predict(x,1) 
			for i in range(1,clip.PElog_spectra.shape[2]):
				x = [] 
				x.append(get_features(clip))
				x = np.array(x)
				clip_prediction *= model.model.predict(x,1)  # multiply probabilities for the segments
			print(np.argmax(prediction),clip.target)	
			print(clip_prediction[0][int(clip.target)])

def evaluate(load_path):
	model = Model()
	model.model.load_weights(load_path+"weights_best.hdf5")
	
	sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
	model.model.compile(optimizer=sgd,loss="categorical_crossentropy", metrics=['accuracy'])

	x = {}
	y = {}
	for fold in clips:
		fold_x = []
		fold_y = []    
		for clip in clips[fold]:
			for i in range(clip.PElog_spectra.shape[2]):
				fold_y.append(keras.utils.to_categorical(clip.target,num_classes=50))
				fold_x.append(get_features(clip))
		x[int(fold)] = np.array(fold_x)
		y[int(fold)] = np.array(fold_y)

	x_test =[]
	y_test =[]

	for _fold in x:
		x_test.extend(x[_fold])
		y_test.extend(y[_fold])
	
	x_test = np.array(x_test)
	y_test = np.array(y_test)	

	results = model.model.evaluate(x=x_test,y=y_test,batch_size=x_test.shape[0],verbose=1)
	print("Test loss = ",results[0])
	print("Test accuracy = ",results[1])

train("./ESC-50-master/model/fold2/",fold=2)
# predict("./ESC-50-master/model/fold1/")
# evaluate("./ESC-50-master/model/fold1/")
