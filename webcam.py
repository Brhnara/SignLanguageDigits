# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:04:31 2019

@author: LENOVO
"""

import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

new_model=load_model('best_model.h5')

capt=cv2.VideoCapture(0)
while True:
    ret,frame=capt.read()
    image=frame[0:300,0:300]
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (100, 100))
    x = img_to_array(img_gray)
    pred=new_model.predict(np.expand_dims(x,axis=0))
    pred=np.argmax(pred)
    cv2.putText(frame,'The digit is '+str(pred), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)    
    
    cv2.imshow('img',frame)
    cv2.imshow('digit',img_gray)
    
    
    if cv2.waitKey(3) & 0xFF==ord('q'):
        break
capt.release()
cv2.destroyAllWindows()