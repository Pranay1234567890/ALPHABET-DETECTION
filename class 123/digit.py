import cv2 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from PIL import Image 
import PIL.ImageOps 
import os, ssl, time 
from scipy.io import loadmat

data1  = loadmat("mnist-original.mat")
x= data1["data"].T
y = data1["label"][0]

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=7000,test_size=3000)
train_x_scale = x_train/255.0
test_x_scale = x_test/255.0

object1 = LogisticRegression(solver='saga', multi_class='multinomial')
object1 = object1.fit(train_x_scale,y_train)

y_result = object1.predict(test_x_scale) 
accuracy = accuracy_score(y_result , y_test) 
print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY)

        height,width = gray.shape
        upper_left = (int(width/2 - 56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2 +56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)

        image_bw =  im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resized_inverted)
        print("working")
        image_bw_resized_inverted_scaled =np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sameple = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_predict = object1.predict(test_sameple)

        print("predited classs is : ",test_predict)

        cv2.imshow('framne',gray)
        if cv2.waitKey(1) == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()