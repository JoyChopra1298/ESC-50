from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.regularizers import l2

class Model:

	def __init__(self):
		self.num_classes = 50						
		self.input_shape = (60,41,2) 				
		self.epochs = 0 							# Useful for model retraining
		self.model = Sequential()
		self.fold = 1								# The fold which is used for cross-vaidation. Rest all are used for training
		
		# First Convolution Layer 
		self.model.add(Conv2D(filters=80,kernel_size=(57,6),activation="relu",kernel_regularizer=l2(0.001),input_shape=self.input_shape))
		self.model.add(MaxPooling2D(pool_size=(4,3), strides=(1,3)))
		self.model.add(Dropout(0.5))
		
		# Second Convolution Layer 
		self.model.add(Conv2D(filters=80,kernel_size=(1,3),activation="relu",kernel_regularizer=l2(0.001),input_shape=self.input_shape))
		self.model.add(MaxPooling2D(pool_size=(1,3), strides=(1,3)))
		
		#Fully Connected Layer
		self.model.add(Flatten())
		self.model.add(Dense(units=500,activation="relu",kernel_regularizer=l2(0.001)))
		self.model.add(Dropout(0.5))
		
		#Output Layer
		self.model.add(Dense(units=self.num_classes,activation="softmax",kernel_regularizer=l2(0.001)))

		#print intermediate data shapes
		print("The shapes of neural net tensors are: (from input to output)")
		for layer in self.model.layers:
			print(layer.output.shape)

######### There is error in PEFBE paper.The Conv net filter for first layer would be (57,6) otherwise can't do max-pooling of (4,3)