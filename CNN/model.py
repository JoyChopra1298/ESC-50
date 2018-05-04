from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.regularizers import l2

def make_model():
		num_classes = 50						
		input_shape = (60,41,2) 				
		model = Sequential()
		
		# First Convolution Layer 
		model.add(Conv2D(filters=80,kernel_size=(57,6),activation="relu",kernel_regularizer=l2(0.001),input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(4,3), strides=(1,3)))
		model.add(Dropout(0.5))
		
		# Second Convolution Layer 
		model.add(Conv2D(filters=80,kernel_size=(1,3),activation="relu",kernel_regularizer=l2(0.001),input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(1,3), strides=(1,3)))
		
		#Fully Connected Layer
		model.add(Flatten())
		model.add(Dense(units=500,activation="relu",kernel_regularizer=l2(0.001)))
		model.add(Dropout(0.5))
		
		#Output Layer
		model.add(Dense(units=num_classes,activation="softmax",kernel_regularizer=l2(0.001)))

		#print intermediate data shapes
		print("The shapes of neural net tensors are: (from input to output)")
		print(model.layers[0].input_shape)
		for layer in model.layers:
			print(layer.output.shape)

		return model	

######### There is error in PEFBE paper.The Conv net filter for first layer would be (57,6) otherwise can't do max-pooling of (4,3)