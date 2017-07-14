

# coding: utf-8

# **Using model with keras**

# In[ ]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import skimage
from skimage import io, transform
warnings.simplefilter("ignore")
import os
# In[ ]:


#saving the path of input images for tarining and testing
train_path='input/train/'
test_path='input/test/'


# In[ ]:


#importing the class labels
y=pd.read_csv('input/train_labels.csv')
print(y.head())


# In[ ]:


#checking for images

im2=io.imread(train_path+'5.jpg')
plt.imshow(im2)


# In[ ]:


#creating numpy arrays for storing images
#here we are taking 200 images for training
#& 100 images for testing
#all images have size of 64x64
X=np.empty(shape=(2295,112,112,3))
y=y.iloc[:2295,1].values


# In[ ]:


#saving image as numpy array
for i in range(0,2295):
    im=io.imread(train_path+str(i+1)+'.jpg')
    X[i]=transform.resize(im,output_shape=(112,112,3))

sample_submission = pd.read_csv("input/sample_submission.csv")
# In[ ]:
skimage.io.reset_plugins()

#creating training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=(0.1),random_state=0)


# In[ ]:


#creating our model model
import keras
from keras.models import Sequential
from keras.layers import Convolution2D,Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model

# In[ ]:

# In[ ]:
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

img_rows, img_cols, img_channel = 112, 112, 3

base_model = applications.VGG19(weights='imagenet',
                                input_shape=(img_rows, img_cols, img_channel),
                                include_top=False)

model = Model(inputs=base_model.input,
                output=base_model.output)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

print(model.summary())

batch_size = 32
epochs = 50

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(X_train)


y_predict=model.predict(X_test)
# In[ ]:
#y_predict
# In[ ]:
#checking the accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
# In[ ]:
scaled=(112,112,3)

# In[ ]:
j=[]
test_names=[]
file_paths = []
img_path = "input/test/"

for i in range(len(sample_submission)):
    test_names.append(sample_submission.ix[i][0])
    file_paths.append( img_path + str(int(sample_submission.ix[i][0])) +'.jpg' )
test_images = []
for file_path in file_paths:
    #read image
    img = io.imread(file_path)
    img = transform.resize(img, output_shape=scaled)
    test_images.append(img)

    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

test_images = np.array(test_images)

# In[ ]:

predictions = model.predict(test_images)
print(predictions)
for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]

sample_submission.to_csv("submit224_bs25.csv", index=False)
