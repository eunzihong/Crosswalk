import numpy as np
import cv2
import os
import scipy.misc

data_path = 'data'
path_of_outputs = "preprocessed_data\\"
data_files = os.listdir(data_path)
out_height = 300
out_width = int(out_height*0.75)

for img_name in data_files:
    load_name = os.path.join(data_path, img_name)
    img = cv2.imread(load_name)

    # resizing
    print((out_width, out_height))
    img = scipy.misc.imresize(img, (out_height, out_width))
    H, W = img.shape[:2]

    #### cut ####
    print(int(0.4*H),H)
    img = img[int(0.4*H):,:]

    #### adjust ####
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray_img)

    # save
    cv2.imwrite(path_of_outputs + img_name+'.png', eq)