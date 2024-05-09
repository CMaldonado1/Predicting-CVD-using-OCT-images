import numpy as np
import sys
import torch.utils.data as data
import torch
import glob
import pandas as pd
import pdb
import sys
from itertools import chain, combinations
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from  sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
from IPython import embed
from PIL import Image
from sklearn.model_selection import StratifiedKFold

def load_img(img_path):
     if img_path is not None:
         oct_array = np.load(img_path, allow_pickle=True)
         oct_img = torch.from_numpy(np.array(oct_array.item().volume).astype('float32'))
         oct_img_resized = F.resize(oct_img,(224,224))
         oct_img_resized = oct_img_resized[0:128] #0:10]   #nlayer]
     else:
         oct_img_resized = None
     return oct_img_resized


class MM(data.Dataset):
    """ Multi-Modal Dataset.
        Args:
        dir_imgs_left (string): Root directory of left OCT images.
        dir_imgs_right (string): Root directory of right OCT images.
        ids_set (pandas DataFrame): DataFrame containing left and right eye data.
    """

    def __init__(self, dir_imgs_left, dir_imgs_right, ids_set):
        self.labels = {}
        self.path_imgs_oct_left = {}
        self.path_imgs_oct_right = {}
        self.mtdt = {}
        self.ids_set = {} #ids_set.reset_index(drop=True)
        self.null_image = torch.zeros(128, 224, 224) #from_numpy(np.full((128, 224, 224), np.nan))
        le = OrdinalEncoder()
        scaler = MinMaxScaler()

        diagnoses = le.fit_transform(ids_set['Vascular'].values.reshape(-1,1))

        mtdt_dataframe = ids_set[['Gender', 'DBP1', 'SBP1', 'BMI', 'Age', 'HbA1c']]
        mtdt_scaled = pd.DataFrame(scaler.fit_transform(mtdt_dataframe), columns=mtdt_dataframe.columns)

        for idx, ID in enumerate(ids_set['SUB_ID'].values):
            imgs_per_id_left = glob.glob(dir_imgs_left + '/*.npy')
            img_oct_left = [j for j in imgs_per_id_left if str(int(ID)) in j]
            # Check if the ID exists in the DataFrame for left eye

            imgs_per_id_right = glob.glob(dir_imgs_right + '/*.npy')
            img_oct_right = [j for j in imgs_per_id_right if str(int(ID)) in j]
            self.ids_set[ID] = str(int(ID))
            self.labels[ID] = diagnoses[idx]
            self.mtdt[ID] = [mtdt_scaled[column][idx] for column in mtdt_scaled.columns]

            if img_oct_left:
               imgs_per_id_left = str(img_oct_left[0])

            self.path_imgs_oct_left[ID] =imgs_per_id_left if img_oct_left else None

            # Check if the ID exists in the DataFrame for right eye

                
            if img_oct_right:
               imgs_per_id_right = str(img_oct_right[0])
            self.path_imgs_oct_right[ID] = imgs_per_id_right if img_oct_right else None

#            self.path_imgs_oct_left = self.path_imgs_oct_left.get(ID, None)   
#            self.path_imgs_oct_right = self.path_imgs_oct_right.get(ID, None)

    # Denotes the total number of samples
    def __len__(self):
        return len(self.ids_set)       #set(self.ids_set_left + self.ids_set_right)) #max(len(self.path_imgs_oct_left), len(self.path_imgs_oct_right))

    # This generates one sample of data
    def __getitem__(self, index):
        """
        Args:
            index (tuple): Index
        Returns:
            tuple: (oct)
        """
        ids = list(self.ids_set.keys())[index % len(self.ids_set)]

        oct_left = load_img(self.path_imgs_oct_left.get(ids, self.null_image))  
        oct_right = load_img(self.path_imgs_oct_right.get(ids, self.null_image))                            #[index]) if self.path_imgs_oct_right[index] else None
        oct_imag_left = (oct_left - torch.min(oct_left))/(torch.max(oct_left) - torch.min(oct_left)) if oct_left is not None else self.null_image
        oct_imag_right = (oct_right - torch.min(oct_right))/(torch.max(oct_right) - torch.min(oct_right)) if oct_right is not None else self.null_image

        label = self.labels.get(ids, None)
        mtdt = self.mtdt.get(ids, None)
        ids_lr = self.ids_set.get(ids, None)
        maskl = torch.ones_like(torch.from_numpy(label)) if oct_imag_left is not self.null_image else torch.zeros_like(torch.from_numpy(label))
        maskr = torch.ones_like(torch.from_numpy(label)) if oct_imag_right is not self.null_image else torch.zeros_like(torch.from_numpy(label))
        return oct_imag_left if oct_imag_left is not None else self.null_image, \
               oct_imag_right if oct_imag_right is not None else self.null_image, \
               torch.LongTensor(np.array([int(label)])), \
               torch.FloatTensor(np.array(mtdt)).float(), \
               ids_lr, \
               maskl if oct_imag_left is not self.null_image else torch.zeros_like(torch.from_numpy(label)), \
               maskr if oct_imag_right is not self.null_image else torch.zeros_like(torch.from_numpy(label))\

#        else:
#           return None

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.n_batches)  #y)


class OCT_DM(pl.LightningDataModule):    
        
    def __init__(self, 
        data_dir_left: Union[None, str] = None,
        data_dir_right: Union[None, str] = None,
 #       data_dir_mi: Union[None, str] = None,
        include_ids: Union[None, List[int], str] = None,
        exclude_ids: Union[None, List[int], str] = None,
        split_lengths: Union[None, List[int]]=None,
        shuffle=False,
        img_size: Union[None, List[int], str] = None,
        batch_size: Union[None, int] = None,  #int = 8,
    ):

        '''
        params:
            data_dir:
            batch_size:
            split_lengths:
        '''
        
        super().__init__()
        self.data_dir_left = data_dir_left
        self.data_dir_right = data_dir_right
        self.include_ids = include_ids
        self.batch_size = batch_size
        self.split_lengths = split_lengths
        self.shuffle = shuffle
        self.img_size = img_size
        
        self.include_ids = pd.read_excel(self.include_ids)

    def my_collate(self, batch):
        len_batch = len(batch)
        batch = list(filter (lambda x:x is not None, batch))
        if len_batch > len(batch):
            diff = len_batch - len(batch)
            for i in range(diff):
                batch = batch + batch[:diff]
        return torch.utils.data.dataloader.default_collate(batch)        

    def setup(self, stage: Optional[str] = None):
        
        #
        # read images
        #
        # TODO: add logic for exclude_ids
        imgs = MM(self.data_dir_left, self.data_dir_right, self.include_ids)
        print('Found ' + str(len(imgs.path_imgs_oct_left)) + ' oct left images')
        print('Found ' + str(len(imgs.path_imgs_oct_right)) + ' oct right images')
#        embed()

#        max_length = max(len(imgs.path_imgs_oct_left), len(imgs.path_imgs_oct_right)) 
        if self.split_lengths is None:
            train_len = int(0 * len(imgs))
            test_len = int(0.99 * len(imgs))
            val_len = len(imgs) - train_len - test_len
            self.split_lengths = [train_len, val_len, test_len]

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(imgs, self.split_lengths)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)
        
#    def predict_dataloader(self):
#        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)        

#    def train_dataloader(self):
#        bacth_sampler_train = []
#        for i in range(len(self.train_dataset)):
#            bacth_sampler_train.append(self.train_dataset[i][1])
#        y_train = [list(x[0].numpy()).index(1.) for x in bacth_sampler_train ]    
#        y_train = [x[0].numpy() for x in bacth_sampler_train ] 
#        return DataLoader(self.train_dataset, num_workers=8, batch_sampler=StratifiedBatchSampler(np.array(y_train),batch_size=self.batch_size), shuffle=self.shuffle, collate_fn=self.my_collate )

#    def val_dataloader(self):
#        bacth_sampler_val = []
#        for i in range(len(self.val_dataset)):
#            bacth_sampler_val.append(self.val_dataset[i][1])
#        y_val = [x[0].numpy() for x in bacth_sampler_val]
#        y_val = [list(x[0].numpy()).index(1.) for x in bacth_sampler_val ]   
#        return DataLoader(self.val_dataset, num_workers=8, batch_sampler=StratifiedBatchSampler(np.array(y_val), batch_size=self.batch_size), shuffle=self.shuffle, collate_fn=self.my_collate) #,drop_last=True)

#    def test_dataloader(self):
#        bacth_sampler_test = []
#        for i in range(len(self.test_dataset)):
#            bacth_sampler_test.append(self.test_dataset[i][1])
#        y_test = [x[0].numpy() for x in bacth_sampler_test ] 
#        y_test = [list(x[0].numpy()).index(1.) for x in bacth_sampler_test ]    
#        return DataLoader(self.test_dataset, num_workers=8, batch_sampler=StratifiedBatchSampler( np.array(y_test), batch_size=self.batch_size  ), shuffle=self.shuffle, collate_fn=self.my_collate) #, drop_last=True)




## For dealing with include and exclude lists

#    def get_indiv_f(self, config):
#
#        """
#          Create a temporal file containing the ID's of the subjects to be included
#          and return its path
#        """
#
#        # TODO: establish default behaviour for when white list is not provided.
#        sample_white_lists = config.get("sample_white_lists", None)
#        sample_black_lists = config.get("sample_black_lists", None)
#
#        wl = []
#
#        if sample_white_lists is not None:
#            for file in sample_white_lists:
#                wl.append(set(pd.read_csv(file, sep="\t").iloc[:,0]))
#            wl = set.intersection(*wl)
#        else:
#            return None
#
#        if sample_black_lists is not None:
#            for file in sample_black_lists:
#                bl.append(set(pd.read_csv(file, sep="\t").iloc[:,0]))
#            bl = set.union(*bl)
#        else:
#            bl = set()
#        
#        wl = wl - bl
#
#        indiv_f = os.path.join(self.tmpdir, "subjects.txt")
#        with open(indiv_f, "w") as indiv_fh:
#          indiv_fh.write("\n".join([str(x) + "\t" + str(x) for x in wl]))
#
#        return indiv_f # config["filename_patterns"].get("individuals", None)
