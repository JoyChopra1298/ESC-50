# ESC-50
Classifiers for the Environmental Sound Classification dataset. 

Directory Structure
```
/				(root directory where you want to store data)
|--audio/ 			(contains all 2000 recordings)
|--category_target.pkl		(Category target dictionary)
|--pickled/			(Contains pickled dataset after using utils.pickle_dataset())	
|--model/
	|--fold1/		(Contains model trained with fold 1 as validation set. Also has log file of the training process.)
	|--fold2/
	|--fold3/
	|--fold4/
	|--fold5/
```
Each fold has a csv file storing training log and .hdf5 storing trained model's weights.

Steps to run -  

1. Get the audio files from ESC-50 github. https://github.com/karoldvl/ESC-50.git
2. Run pickle_dataset from utils.py. It will compute the features for each clip and store them
3. Then run train from CNN.py for each fold.
4. Run evaluate from CNN.py to evaluate the models.

Source Code -
1. clip.py - Contains class Clip for audio clip. Features such as Phase encoded Mel Filterbank Energies (PEFBEs) and Filterbank Energies (FBEs) are extracted here.
2. model.py - Contains Model class which makes the CNN model's architechture.
3. main.py - Contains the functions to train,predict and evaluate the model.
4. utils.py - Contains functions for saving and loading the dataset.
