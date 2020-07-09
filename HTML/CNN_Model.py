#!/usr/bin/env python
# coding: utf-8

# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten,BatchNormalization,Activation,Conv2D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping 
from keras import optimizers, regularizers, Model
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.callbacks import ModelCheckpoint

# Utility
import os
import pandas as pd
from os import listdir
from os.path import isfile, join 
import numpy as np
import itertools
import random
from collections import Counter
from glob import iglob
import seaborn as sn
import pickle

# Import OpenCV
import cv2
 
# Ignore warning
import warnings
warnings.filterwarnings('ignore')


# ### Model Setting

# In[57]:


IMAGE_SIZE = (256,256)
INPUT_SHAPE = (256,256,3)

BATCH_SIZE = 12
EPOCHS = 20
STEP_PER_EPOCHS = 50


# ### Read Image Dataset

# In[58]:


base_dir = 'dataset_v2'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
predict_dir = os.path.join(base_dir, 'predict')


# ### Creating objects for image augmentations

# In[59]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip=True,
                                   rotation_range=90)
 
test_datagen = ImageDataGenerator(rescale=1./255)

 
train_generator = train_datagen.flow_from_directory(train_dir,
                            target_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE,
                            class_mode='categorical')
 
test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)


# ### Setup model layers

# In[60]:


model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = INPUT_SHAPE, activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())

# Adding a third convolutional layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())

# Adding a fourth convolutional layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
model.add(Flatten())   
# Step 4 - Full connection
model.add(BatchNormalization())
model.add(Dense(units = 1024, activation = 'relu'))

model.add(Dense(units = 5, activation = 'softmax'))
 


# In[61]:


optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# ### Fit Model

# In[62]:


STEP_PER_EPOCHS = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = test_generator.n//test_generator.batch_size

model_history = model.fit(train_generator,steps_per_epoch= STEP_PER_EPOCHS,
                    epochs=EPOCHS, validation_data=test_generator, validation_steps=STEP_SIZE_VALID,
                    callbacks=[EarlyStopping(monitor='loss', patience=5)])


# In[63]:


model.summary()


# ### Model Accuracy and Loss Plots

# In[64]:


plt.rcParams['figure.figsize'] = (12.0, 6.0)

def model_performance_plot(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],label='Training Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],label = 'Validation Accuracy')
    plt.legend()
  

    plt.figure()
    plt.xlabel('Epochs')
    plt.title('Training and Aalidation Categorical Crossentropy Loss')
    plt.plot(hist['epoch'], hist['loss'],label='Training Loss')
    plt.plot(hist['epoch'], hist['val_loss'],label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


model_performance_plot(model_history)


# ### Evaluate Test Dataset

# In[65]:


loss, accuracy = model.evaluate_generator(test_generator, steps=test_generator.samples//test_generator.batch_size)


# In[66]:


print("Test dataset Accuracy: %f and Loss: %f" % (accuracy,loss))


# ### Model Classification and Confusion Matrix

# In[67]:


Y_pred = model.predict_generator(test_generator, len(test_generator))
y_pred = np.argmax(Y_pred, axis=1)


# In[68]:


target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred, target_names=target_names)) 


# In[69]:


conf_mat = confusion_matrix(test_generator.classes, y_pred)
df_cm = pd.DataFrame(conf_mat, index = target_names, columns = target_names)
plt.figure(figsize = (12,8))
sn.heatmap(df_cm, annot=True,cmap="YlGnBu")


# ### Predict New Image

# In[70]:


class_mapping = {v:k for k,v in train_generator.class_indices.items()}


# In[73]:


def image_type(filename):
    
    if "glass" in filename:
        return 'glass'
    elif "metal" in filename:
        return 'metal'
    elif "paper" in filename:
        return 'paper'
    elif "plastic" in filename:
        return 'plastic'
    elif "cardboard" in filename:
        return 'cardboard'
    

def plot_predicted_label(nrows, ncols,onlyfiles):
    fig, axes = plt.subplots(nrows, ncols,figsize=(20, 13))
    
    rs = np.random.choice(len(onlyfiles),nrows*ncols,replace=False)
    
    for i, ax in zip(rs, axes.flat):
        img = cv2.imread(os.path.join(predict_dir, onlyfiles[i]))
        true_type = image_type(onlyfiles[i])
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1])) 
        img = img /255
         
        probabilities = model.predict(np.asarray([img]))[0]
        class_idx = np.argmax(probabilities)
            
        title = 'True: %s, Pred: %s , Confi:%0.2f' % (true_type,class_mapping[class_idx],probabilities[class_idx])
        ax.imshow(img, cmap='binary')
        ax.set_title(title)
         
        ax.set_xticks([])
        ax.set_yticks([])

        
num_rows = 2
num_cols = 4

# load image path to predict
onlyfiles = [f for f in listdir(predict_dir) if isfile(join(predict_dir, f))]

plot_predicted_label(num_rows, num_cols,onlyfiles);


# ### Save Model

# In[72]:


model.save('CNNModel.h5')

