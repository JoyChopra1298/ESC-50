import os
import pickle
from clip import Clip

def pickle_dataset(path,only_one=False):
    """Saves all Clip instances to path/pickled directory.Need to be done only once"""
    clips = []
    audio_path = path+"/audio"
    category_path = path+"/category_target.pkl"
    pickling_path = path+"/pickled/"
    category_dict = pickle.load(open(category_path,"rb"))
    
    i = 0
    pickling_interval = 400 # so each fold stored separately
    os.makedirs(pickling_path, exist_ok=True)

    if(only_one):
    #Saves only one pkl. Useful for testing purposes    
        for wav in sorted(os.listdir('{0}/'.format(audio_path))):
            clip = Clip(audio_path,wav)
            print(clip.target)
            print("Category: "+category_dict[clip.target]+" wav file "+wav)
            clip.category = category_dict[clip.target]
            clips.append(clip)
            i+=1
            if(i%pickling_interval==0):  
              pickle.dump(clips,open(pickling_path+"fold{0}.pkl".format(int(i/pickling_interval)),"wb"))
              clips = []
              print("-------------Pickled clips from {0} to {1}---------".format(i-pickling_interval+1,i))
              break
            print(clip) 
    else: 
        for wav in sorted(os.listdir('{0}/'.format(audio_path))):
            clip = Clip(audio_path,wav)
            print(clip.target)
            print("Category: "+category_dict[clip.target]+" wav file "+wav)
            clip.category = category_dict[clip.target]
            clips.append(clip)
            i+=1
            if(i%pickling_interval==0):
                pickle.dump(clips,open(pickling_path+"fold{0}.pkl".format(int(i/pickling_interval)),"wb"))
                clips = []
                print("-------------Pickled clips from {0} to {1}---------".format(i-pickling_interval+1,i))
            print(clip)  

def load_dataset(pickle_path,only_one=False):
    """Loads and returns dataset clips from pickled_path"""
    clips = {}
    if(only_one):
        #Loads only one pkl. Useful for testing purposes
        pkl = sorted(os.listdir(pickle_path))[0]
        clips["1"] = pickle.load(open(pickle_path+"/"+pkl,"rb"))
        print("loaded clips from ",pkl)
    else:
        fold = 1
        for pkl in sorted(os.listdir(pickle_path)):
            clips["{0}".format(fold)] = pickle.load(open(pickle_path+"/"+pkl,"rb"))
            print("loaded clips from ",pkl)
            fold+=1
    return clips

if __name__=="__main__":

    pickle_dataset("./ESC-50-master")   # (NOTE This creates a file of size in GBs.)

    # clips = load_dataset("./ESC-50-master/pickled",only_one=True)
    # clips = load_datsaset("./ESC-50-master/pickled")
    