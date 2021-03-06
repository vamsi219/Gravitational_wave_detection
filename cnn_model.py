#Importing the required libraries

from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.metrics import AUC
import torch
from nnAudio.Spectrogram import CQT1992v2

#Forming the Dataframe of Train data, and adding PATH column to it.
signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo")
train_data = pd.read_csv('ing_labels.csv')
paths_lst =[]
lst = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']
for i in range(16):
    for j in range(16):
        for k in range(16):
            All_NPY_Path = Path(str(lst[i]) + "/" + str(lst[j]) + "/" +str(lst[k]))
            PY_Path_List = list(All_NPY_Path.glob(r"*.npy"))
            PY_Path_List.sort()
            paths_lst.extend(PY_Path_List)
train_data["PATH"] = paths_lst

print("-"*100)
print("The dataframe formed using the given data is: ")
print("-"*100)
print(train_data.head(5))
print("-"*100)
#--------------------------------------------------------------------------------------------------------------------
#DATA EXPLORATION

#Seeing Number of 1's and 0's
sns.countplot(data = train_data, x="target")
print(train_data['target'].value_counts())
print("-"*100)

#Let's see how each obersvation in the data lokk like
sample = np.load(paths_lst[1])
print("Data lokks like: ",sample)
print("-"*100)
print("We can see that there are 3 rows to the data. This represents data extracted by 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo) respectively.")
print("-"*100)
print("Data in the signal format")
print("-"*100)
plt.plot(sample[0])
plt.plot(sample[1])
plt.plot(sample[2])
plt.legend(signal_names, fontsize=12, loc="lower right")
print("-"*100)

#DATA PREPROCESSING


# CQT
transform = CQT1992v2(sr=2048,fmin=20,fmax=500,hop_length=64,verbose=False)

# preprocess-cqt function
def preprocess_cqt(path):
    sig = np.load(path.numpy())
    for i in range(sig.shape[0]):     
        sig[i] /= np.max(sig[i])             # normalize signal
    sig = np.hstack(sig)                     # horizontal stack
    sig = torch.from_numpy(sig).float()      # tensor conversion
    image = transform(sig)                   # getting the image from CQT transform
    image = np.array(image)                  # converting to array from tensor
    image = np.transpose(image,(1,2,0))      # transpose the image to get right orientation
    return tf.convert_to_tensor(image)       # conver the image to tf.tensor and return

input_shape = (56, 193, 1)

def preprocess_function_parse_tf(path, y=None):
    [x] = tf.py_function(func=preprocess_cqt, inp=[path], Tout=[tf.float32])
    x = tf.ensure_shape(x, input_shape)
    if y is None:
        return x
    else:
        return x,y

# Forming Data sets and Target sets
X = train_data["id"]
y = train_data["target"].astype('int8').values

batch_size = 280

#Splitting Data into Train and Validation Datasets
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state = 42, stratify = y)

#File path function
def get_npy_filepath(id_, is_train=True):
    if is_train:
        return f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}.npy'
    else:
        return f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}.npy'

#TRAIN DATASET AND VALIDATION DATASET
train_dataset = tf.data.Dataset.from_tensor_slices((x_train.apply(get_npy_filepath).values, y_train))
train_dataset = train_dataset.shuffle(len(x_train))
train_dataset = train_dataset.map(preprocess_function_parse_tf, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid.apply(get_npy_filepath).values, y_valid))
valid_dataset = valid_dataset.map(preprocess_function_parse_tf, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

#CREATING THE MODEL
train_dataset.take(1)                    
model_cnn = Sequential(name='CNN_model') 

# Add the first Convoluted2D layer w/ input_shape & MaxPooling2D layer followed by that
model_cnn.add(Conv2D(filters=16, kernel_size=3, input_shape=input_shape, activation='relu', name='Conv_01'))
model_cnn.add(MaxPooling2D(pool_size=2, name='Pool_01'))

# Second pair of Conv1D and MaxPooling1D layers
model_cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu', name='Conv_02'))
model_cnn.add(MaxPooling2D(pool_size=2, name='Pool_02'))

# Third pair of Conv1D and MaxPooling1D layers
model_cnn.add(Conv2D(filters=64, kernel_size=3, input_shape=input_shape, activation='relu', name='Conv_03'))
model_cnn.add(MaxPooling2D(pool_size=2, name='Pool_03'))

# Add the Flatten layer
model_cnn.add(Flatten(name='Flatten'))

# Add the Dense layers
model_cnn.add(Dense(units=512, activation='relu', name='Dense_01'))
model_cnn.add(Dense(units=64, activation='relu', name='Dense_02'))

# Add the final Output layer
model_cnn.add(Dense(1, activation='sigmoid', name='Output'))

model_cnn.summary()
print("-"*100)

model_cnn.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy', metrics=[[AUC(), 'accuracy']])
# Fit the data
history_cnn = model_cnn.fit(x=train_dataset, epochs=100,validation_data=valid_dataset,batch_size=batch_size,verbose=1)

prediction = model_cnn.predict(valid_dataset)
prediction = prediction.flatten()
# predict
y_pred = list(prediction)

# f1 score
precision_score = precision_score(y_pred, y_valid)
print("precsion score is",precision_score)

recall_score = recall_score(y_pred, y_valid)
print("recall score is",recall_score)

f1_score = f1_score(y_pred, y_valid)
print("f1 score is",f1_score)

#We can use this model for predciting the traget values of test dataset