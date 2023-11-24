import os
import cv2
import numpy as np
from ultralytics import YOLO

path=os.path.dirname(__file__)
path=path+''
os.chdir(path)

model = YOLO("best.pt",task='detect')

for i in os.listdir(path+'/test_imgs/'):
    img=path+'/test_imgs/'+i
    image=cv2.imread(img)
    Box=[]
    result=model(image,imgsz=800)
    for box in result[0].boxes:
        conf=box.conf[0].cpu().detach().numpy().tolist()
        if Box!=[]:
            if conf<Box[5]:
                continue
        num=box.cls[0].cpu().detach().numpy().tolist()
        box=box.xyxy
        x1=int(box[0][0].cpu().detach().numpy().tolist())
        y1=int(box[0][1].cpu().detach().numpy().tolist())
        x2=int(box[0][2].cpu().detach().numpy().tolist())
        y2=int(box[0][3].cpu().detach().numpy().tolist())
        w=x2-x1
        h=y2-y1
        Box=[x1,y1,w,h,num,conf]
    rec=(Box[0],Box[1],Box[2],Box[3])
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(image,mask,rec,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    image = image * mask_2[:, :, np.newaxis]
    image=image[y1:y2,x1:x2]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    size=(800,800)
    x=x2-x1
    y=y2-y1
    base=np.zeros((size[1],size[0]),np.uint8)
    w=size[0]/x
    h=size[1]/y
    if h<w:
        re_size=(int(x*h),int(y*h))
    else:
        re_size=(int(x*w),int(y*w))
    re_image = cv2.resize(image,dsize=re_size)
    base[int(size[1]/2-re_size[1]/2):int(size[1]/2+re_size[1]/2),int(size[0]/2-re_size[0]/2):int(size[0]/2+re_size[0]/2)]=re_image
    #image=cv2.equalizeHist(base)
    #cv2.imshow('',image)
    #if cv2.waitKey(0) & 0xFF == ord("q"):
    #    continue
    cv2.imwrite(path+"/grab_cut_imgs/"+i,base)