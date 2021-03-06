from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
# from tensorflow.keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2


train_dir = './dataset/asl-alphabet_exclude/asl_alphabet_train/asl_alphabet_train'
test_dir = './dataset/asl-alphabet_exclude/asl_alphabet_test/asl_alphabet_test'

# **Plotting one image from each class :**
def load_unique():
    size_img = 64, 64
    images_for_plot = []
    labels_for_plot = []
    for folder in os.listdir(train_dir):
        for file in os.listdir(train_dir + '/' + folder):
            filepath = train_dir + '/' + folder + '/' + file
            print(filepath)
            image = cv2.imread(filepath)
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            images_for_plot.append(final_img)
            labels_for_plot.append(folder)
            break
    return images_for_plot, labels_for_plot


images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot)

# **LOADING DATA**



# 5. creating training and test data by splitting original data into 95% of training data and 5% testing data.

# 1. Defining a dictionary which contains labels and its mapping to a number which acts as class label.;
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
               'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
               'Z': 25, 'space': 26, 'del': 27, 'nothing': 2}




# 2. loading image data and labels and then mapping those labels to the dictionary defined before.
def load_data():
    """
    Loads data and preprocess. Returns train and test data along with labels.
    """
    images = []
    labels = []
    size = 64, 64
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(labels_dict[folder])

    images = np.array(images)
    # 3. Normalizing image data.
    images = images.astype('float32')/255.0
    # 4. converting labels to one hot format.
    labels = keras.utils.to_categorical(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(
        images, labels, test_size=0.05)

    print('Loaded', len(X_train), 'images for training,',
          'Train data shape =', X_train.shape)
    print('Loaded', len(X_test), 'images for testing',
          'Test data shape =', X_test.shape)

    return X_train, X_test, Y_train, Y_test



X_train, X_test, Y_train, Y_test = load_data()


# **DEFINING MODEL**

def create_model():

    model = Sequential()

    model.add(Conv2D(16, kernel_size=[
              3, 3], padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=[3, 3],
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=[3, 3]))

    model.add(Conv2D(32, kernel_size=[3, 3],
                     padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3],
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=[3, 3]))

    model.add(Conv2D(128, kernel_size=[3, 3],
                     padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=[3, 3],
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=[3, 3]))

    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(29, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

    print("MODEL CREATED")
    model.summary()

    return model


def fit_model():
    model_hist = model.fit(X_train, Y_train, batch_size=64,
                           epochs=5, validation_split=0.1)
    return model_hist

# **COMPILING THE MODEL AND FITTING THE TRAINING DATA**



model = create_model()
curr_model_hist = fit_model()

# **EVALUATING THE TRAINED MODEL ON THE TEST DATA SPLIT **

evaluate_metrics = model.evaluate(X_test, Y_test)


# Loading the test data Provided.
# * The test data contains images with the image name as the class, to which the image belongs to.
# * we will load the data and check if the predition is correct for the image.

def load_test_data():
    images = []
    names = []
    size = 64, 64
    for image in os.listdir(test_dir):
        temp = cv2.imread(test_dir + '/' + image)
        temp = cv2.resize(temp, size)
        images.append(temp)
        names.append(image)
    images = np.array(images)
    images = images.astype('float32')/255.0
    return images, names


test_images, test_img_names = load_test_data()



# make predictions on an image and append it to the list (predictions).
predictions = [model.predict_classes(image.reshape(1, 64, 64, 3))[0] for image in test_images]



def get_labels_for_plot(predictions):
    predictions_labels = []
    for i in range(len(predictions)):
        for ins in labels_dict:
            if predictions[i] == labels_dict[ins]:
                predictions_labels.append(ins)
                break
    return predictions_labels




