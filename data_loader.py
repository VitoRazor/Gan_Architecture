import numpy as np 
import scipy
from glob import glob
import random
import cv2

class DataLoader():
    def __init__(self, dataset_name, img_res=(64,64,1)):
        self.dataset_name=dataset_name
        self.img_res=img_res
    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        #glob.glob()-----list all file paths
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        batch = np.random.choice(path, size=batch_size)

        imgs_A=[]
        for img in batch:
            img=self.imread(img)
            img_A=cv2.resize(img,self.img_res,interpolation=cv2.INTER_CUBIC)  
            
            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
            imgs_A.append(img_A)
        return imgs_A

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        # shuffle the datasets

        random.shuffle(path)
        self.n_batches=int(len(path)/batch_size)
        while True:
            for i in range(self.n_batches-1):
                batch = path[i*batch_size:(i+1)*batch_size]
                imgs_A = []
                for img in batch:
                    img=self.imread(img)
        
                    img_A=cv2.resize(img, self.img_res[0:2], interpolation=cv2.INTER_CUBIC) 
                    if not (self.img_res[2] > 1):
                        img_A=np.expand_dims(img_A,axis=2)
                    if not is_testing and np.random.random() > 0.5:
                            img_A = np.fliplr(img_A)
                    imgs_A.append(img_A)
                imgs_A= np.array(imgs_A)/127.5-1.
                yield imgs_A

    def load_img(self, path):
        img = self.imread(path)
        img_A=cv2.resize(img,self.img_res,interpolation=cv2.INTER_CUBIC)  

        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        if (self.img_res[2] > 1):
            return cv2.imread(path).astype(np.float)
        else:
            a=cv2.imread(path,0).astype(np.float)
            return np.expand_dims(a,axis=2)