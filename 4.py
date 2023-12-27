import os
import cv2
import numpy as np

path=os.path.dirname(__file__)
path=path+''
os.chdir(path)

for i in os.listdir(path+'/grab_cut_imgs2/'):
    img=path+'/grab_cut_imgs2/'+i
    image=cv2.imread(img)
    
    for i in range(800):
        for j in range(800):
            if np.any(image[i][j])!=0:
                image[i][j]=[255,255,255]
                
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))           
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, k)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, k)
                
    cv2.imshow('',image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
       continue