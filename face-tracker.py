from matplotlib import pyplot as plt
from bob.ip.flandmark import Flandmark
from bob.ip.draw import box, cross
from bob.ip.color import rgb_to_gray
import math

# Importing our packages
import os
import cv2
import numpy as np
from time import time
import numpy as np
#import inspect

def get_data(f):
  from os.path import join
  from pkg_resources import resource_filename
  from bob.io.base import load
  import bob.io.image
  return load(resource_filename('bob.ip.flandmark', join('data', f)))

def get_landmarks(lena_gray, x, y, width, height):
    keypoints = None
    p0 = None

    face = face_classifier.detectMultiScale(lena, 1.2, 5)
    if len(face) == 0:
      return p0,False
      
    # Histogram equalization to improve contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lena_gray = clahe.apply(lena_gray)
    localizer = Flandmark()
    keypoints = localizer.locate(lena_gray, y, x, height, width)
    keypoints = np.fliplr(keypoints)

    i = 0
    p0_len = keypoints.shape[0]
    main = np.array([[[]]])
    while i < p0_len:
      to_add = keypoints[i]
      if i == 0:
        main = to_add[None,:][None,:]
      else:
        main = np.concatenate((main, to_add[None,:][None,:]))
        
      i = i + 1
    main = np.array(main, dtype='f')
    p0 = main

    return p0, True

def get_dimensions_realtime(lena, face_classifier):
  face = face_classifier.detectMultiScale(lena, 1.2, 5)
  x = 0
  y = 0
  width = 0
  height = 0

  #if len(face) == 0:
    #print 'No face was detected'
    #exit()
  #else:
    #print 'Face detected'

  for (x,y,width,height) in face:
      lena_small = lena[y: y+height, x: x+width]
  return lena_small, x, y, width, height

def get_dimensions(lena, face_classifier):
  face = face_classifier.detectMultiScale(lena, 1.2, 5)
  x = 0
  y = 0
  width = 0
  height = 0


  for (x,y,width,height) in face:
      lena_small = lena[y: y+height, x: x+width]
  return lena_small, x, y, width, height

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

'''
cv2.imshow('Testing',img)
cv2.waitKey(0)
'''

#x, y, width, height = [214, 202, 183, 183] #or from OpenCV
face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

begin = cv2.VideoCapture(0)

ret, lena = begin.read()
begin.release()

old_lena = lena.copy()

lena_small, x, y, width, height = get_dimensions(lena, face_classifier)



cv2.imshow('Detected faces', lena_small)
cv2.waitKey(0)

lena_t = lena.copy()

lena_t_gray = cv2.cvtColor(lena_t,cv2.COLOR_BGR2GRAY)

localizer0 = Flandmark()
keypoints0 = localizer0.locate(lena_t_gray, y, x, height, width)

#print 'Keypoints0: ',keypoints0
# Experiment
#keypoints0 = np.fliplr(np.fliplr(keypoints0))
###

for k0 in keypoints0:
  x_cen = int(k0[1])
  y_cen = int(k0[0])
  
  cv2.circle(lena_t,(x_cen,y_cen), 3, (0,0,255), -1)
#print 'Exit loop'

cv2.imshow('Inital Image',lena_t)
cv2.waitKey(0)


lena_gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)

p0, lm_detected = get_landmarks(lena_gray, x, y, width, height)


'''
#print 'p0: ', p0
print 'p0 type: ', type(p0)
print 'p0 shape: ', p0.shape
print 'p0[0,0,0]: ', p0[0,0,0]
print 'type(p0[0,0,0]): ', type(p0[0,0,0])
'''


j = 0
z = 0
cap = cv2.VideoCapture(0)

lk_on = False
while True:
  time0 = time()
  ret, lena = cap.read()  
  
  if cv2.waitKey(1) & 0xFF==ord('q'):
    break
  
  lena_gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
  # p1 is good

  
  
  if j > 10:
    #print 'entering'
    face = face_classifier.detectMultiScale(lena, 1.5, 6)
    if len(face) != 0:
      lena_small, x, y, width, height = get_dimensions_realtime(lena, face_classifier)
      
      p0, lm_detected = get_landmarks(lena_gray, x, y, width, height)

      if lm_detected:
        z = 0
        j = 0
    #else:
      #print 'No face detected'
    
  if z == 0 and lm_detected:
    p1, st, err = cv2.calcOpticalFlowPyrLK(lena, lena, p0, None, **lk_params)
  elif lm_detected:
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_lena, lena, p0, None, **lk_params)
  else:
    #print 'Trying again'
    continue
    j = j + 1
  old_lena = lena.copy()
  
  good_new = p1[st==1]
  good_old = p0[st==1]


  
  if z == 0:
    z = 1
  
  

  #print z 
  for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    #c,d = old.ravel()
    #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    '''
    if z == 0:
      cv2.circle(lena_gray,(b, a),5,color[i].tolist(),-1)
    else:
      cv2.circle(lena_gray,(a, b),5,color[i].tolist(),-1)
    '''

    # Make lucas kanade points visible
    if cv2.waitKey(1) & 0xFF==ord('o'):
      if lk_on:
        lk_on = False
      else:
        lk_on = True

    if lk_on:
      cv2.circle(lena,(a, b),5,color[i].tolist(),-1)

    
    #print i
    #print color[i]
    a,b = new.ravel()
    c,d = old.ravel()


    x_max = np.amax(good_new[:,0])
    #print x_max
    x_min = np.amin(good_new[:,0])
    #print x_min
    y_max = np.amax(good_new[:,1])
    #print y_max
    y_min = np.amin(good_new[:,1])
    #print y_min

    x_dis = int(math.floor(x_max - x_min))
    y_dis = int(math.floor(y_max - y_min))

    x_rad = int(math.floor(x_dis/2))
    y_rad = int(math.floor(y_dis/2))

    x_mid = int(math.floor(x_min + x_rad))
    y_mid = int(math.floor(y_min + y_rad))

    cv2.circle(lena, (x_mid, y_mid), max(x_rad + 35, y_rad + 35), (0,255,0), 10)


    
  # To draw template of aligned position
  '''
  for k0 in keypoints0:
    x_cen = int(k0[1])
    y_cen = int(k0[0])
    
    cv2.circle(lena,(x_cen,y_cen), 3, (0,0,255), -1)
  '''
    
  '''
  lena_gray = lena_gray[y: y+height, x: x+width]

  print lena_gray.shape
  print 'Height: ', height-y
  print 'Width: ', width-x
  '''

  '''
  localizer = Flandmark()
  #keypoints = localizer.locate(lena_gray, 0, 0, lena_gray.shape[0], lena_gray.shape[1])
  keypoints = localizer.locate(lena_gray, y, x, height, width)
  '''
  
  '''
  for k in keypoints:
    x_key = int(k[1])
    y_key = int(k[0])
    cv2.circle(lena,(x_key,y_key), 3, (0,0,255), -1)
  '''

  # Show gcontrasted grayscale video
  cv2.imshow('Processed Image',lena)
  #cv2.imshow('Processed Image',lena_gray[y: y+height, x: x+width])
  
  #cv2.imshow('Processed Image',lena[y: y+height, x: x+width])
  #cv2.waitKey(0)

  
  p0 = good_new.reshape(-1,1,2)
  
  time1 = time()
  '''
  print 'Time passed: ', time1 - time0
  '''
  j = j + 1
cap.release()






