from keras.models import load_model
import cv2
import numpy as np
 
model = load_model('pneumonia_pred_new.h5')
'''
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
'''
imageee = 'your_image.jpeg'
img = cv2.imread(imageee)
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
 
classes = model.predict_classes(img)
probabilities = model.predict_proba(img)
print(classes)
 
from google.colab.patches import cv2_imshow
cvimg = cv2.imread(imageee)
cv2_imshow(cvimg)
if classes == [[1]]:
  pred = 'POSITIVE'
else:
  pred = 'NEGATIVE'
  probabilities = 1 - probabilities
 
 
 
print("------------PREDICTION--------------")
print()
print("PNEUMONIA TEST RESULT : ",pred)
print('The probability of the test being {} is {}% '.format(pred,int(probabilities*100)))
print("------------------------------------")