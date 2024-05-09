import numpy as np
from PIL import Image
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt
from python.filetypes import FDS
#from oct_converter.readers import FDS
import cv2
from IPython import embed


#################### Load OCT #############################
def load_img_oct(img_path):
     fds = FDS(img_path)
     oct_volume = fds.read_oct_volume()
     return oct_volume

#################### Load fundus #############################
def load_img_fundus(img_path):
     fds = FDS(img_path)
     fundus_image = fds.read_fundus_image()
     fundus_img = fundus_image
     return fundus_img



if __name__ == '__main__':

        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--ids_file", default=None)  # path to IDs patiente excel file              #./input_data/df_ID.csv")
        parser.add_argument("--dir_imgs", default=None)  # path to FDS files
        parser.add_argument("--dir_npy", default=None)  # path to npy files
        
        args = parser.parse_args()

        
        IDs = pd.read_csv(args.ids_file, sep=" ", header=None)
        ids_set = IDs[0].copy()
        dir_imgs = args.dir_imgs
        dir_npy = args.dir_npy
        path_imgs_oct = []
        path_imgs_fundus = []
        ids_que_si = []

        for idx, ID in enumerate(ids_set.values):
           
            # Reading all oct images per patient
            imgs_per_id = glob.glob(dir_imgs + '/*.fds')
            img_per_fds = [j for j in imgs_per_id if str(int(ID)) in j]
            if img_per_fds:
                imgs_per_id = str(img_per_fds[0])
                # path for oct images
                path_imgs_oct.append(imgs_per_id)
                # path for oct images
                ids_que_si.append(ID)  
            else:
                print("idx", idx)
                print("dir_imgs", dir_imgs)
                print("img_per_fds",img_per_fds )
                print("ID", ID)
                continue
            # Loading oct image
            oct = load_img_oct(path_imgs_oct[-1])
            np.save(dir_npy + '/oct_{0}.npy'.format(ID), oct)
