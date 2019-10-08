import os
import numpy as np
import cv2
from numpy import loadtxt
from keras.models import load_model
import json 
from keras.models import model_from_json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from shutil import copyfile


def load_data(force=False):
    if not(os.path.exists('y.npy') and os.path.exists('y.npy')) or force:
        x_train, y_train, filenames =  compile_dataset(dataset_dir)
        
        np.save(variables_dir + '/x',x_train)
        np.save(variables_dir + '/y',y_train)
        np.save(variables_dir + '/names',filenames)
        

    else:
        x_train = np.load(variables_dir + '/x.npy')
        y_train = np.load(variables_dir + '/y.npy')
        filenames = np.load(variables_dir + '/names.npy')



    return x_train,y_train, filenames

def split_data(x,y,files,percent_test=20,percent_total=100):
    l = len(x)

    l_scaled = int(percent_total/100*l)
    l_test = int(percent_test/100*l_scaled)
    
    x_train =       x    [0              :l_scaled-l_test-1]
    y_train =       y    [0              :l_scaled-l_test-1]
    files_train =   files[0              :l_scaled-l_test-1]
    
    
    x_test =        x    [l_scaled-l_test:l_scaled]
    y_test =        y    [l_scaled-l_test:l_scaled]
    files_test =    files[l_scaled-l_test:l_scaled]

    return (x_train,y_train,files_train),(x_test,y_test,files_test)

def compile_dataset(dataset_dir,grayscale=False):
    data_files = os.listdir(dataset_dir)

    L = len(data_files)
    if grayscale:
        x_train = np.zeros((L,200,200,1))
    else:
        x_train = np.zeros((L,200,200,3))

    y_train = np.zeros((L,6))
    filenames = (["" for x in range(L)])
    
    num_data = 0

    print("Categorizing filenames")
    for file_name in data_files:
        try:
            y_train_i = category_names.index(file_name[:2])
        except:
            print("Could not categorize filename: " + file_name)
            continue 

        if "bmp" in file_name[-3:]:
            path = dataset_dir + "/" + file_name
            if grayscale:
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #8 bit
            else:
                img = cv2.imread(path)

            x_train[num_data] = img
            y_train[num_data][y_train_i] = 1
            filenames[num_data] = path

            num_data = num_data + 1
            """
            cv2.imshow(category_names[y_train_i]+','+file_name,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

        else:
            print("Incorrect filetype: " + file_name)
            continue

    print("Compiled " + str(num_data+1) + " data points.")    
    x_train = (x_train[:num_data])
    y_train = (y_train[:num_data])
    filenames = (filenames[:num_data])    
    
    return (x_train, y_train, filenames)


def ML(x_train,y_train,x_test,y_test,num_epochs=1):

    #x_train = x_train.reshape(x_train.shape[0], 200, 200, 3)
    x_train = x_train.astype('float32')/255.0
    
    # parameters for architecture
    input_shape = (200, 200, 3)
    num_classes = 6
    conv_size = 32

    # parameters for training
    batch_size = 32

    # load MobileNet from Keras
    MobileNet_model = MobileNet(include_top=False, input_shape=input_shape)

    # add custom Layers
    x = MobileNet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    Custom_Output = Dense(num_classes, activation='softmax')(x)

    # define the input and output of the model
    model = Model(inputs = MobileNet_model.input, outputs = Custom_Output)
            
    # compile the model
    model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=0.1)

    model_json = model.to_json()
    with open(variables_dir + "/model_in_json.json", "w") as json_file:
        json.dump(model_json, json_file)

    model.save_weights(variables_dir + "/model_weights.h5")

    print("Saved model to disk")

def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

def clean_output_path(output_dir,incorrect_dir):
    create_dir(output_dir)
    create_dir(output_dir + '/' + incorrect_dir)

    for category in category_names:
        create_dir(output_dir + '/' + category)
        for filename in os.listdir(output_dir + '/' + category):
            os.remove(output_dir + '/' + category + '/' + filename)
    
    for filename in os.listdir(output_dir + '/' + incorrect_dir):
        os.remove(output_dir + '/' + incorrect_dir + '/' + filename)

    


def eval_model(x_test,y_test,files_test,sort = False,output_dir = 'Output',incorrect_dir='Mislabeled'):    
    if sort:
        clean_output_path(output_dir,incorrect_dir)

    with open(variables_dir + '/model_in_json.json','r') as f:
        model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights(variables_dir + '/model_weights.h5')

    model.compile(optimizer='adam', \
            loss='sparse_categorical_crossentropy', \
            metrics=['accuracy'])

    x_test = x_test.astype('float32')/255.0
    y = model.predict(x_test, verbose=2)
    correct = 0
    for i in np.arange(len(y)):
        if np.argmax(y[i]) == np.argmax(y_test[i]):
            img_name = files_test[i].split('/')[-1]
            correct = correct + 1
            if sort:
                copyfile(files_test[i], \
                    output_dir + '/' +\
                    img_name[:2] + '/' +\
                    img_name)
        else:
            print("Mis-categorized " + files_test[i])
            if sort:
                copyfile(files_test[i], \
                    output_dir + '/' +\
                    incorrect_dir + '/' +\
                    img_name)

    print(str(round(correct/len(y),3)*100.0) + "% Correct")

if __name__ == "__main__":
    category_names = ['RS','Pa','Cr','PS','In','Sc']
    dataset_dir = "./NEU_Dataset", variables_dir = "./Parameters"

    x,y,files = load_data(force=True)    
    
    (x_train,y_train,files_train),(x_test,y_test,files_test), = split_data(x,y,files,percent_test = 10,percent_total = 10)
    ML(x_train,y_train,x_test,y_test,num_epochs = 1)
    
    eval_model(x_test,y_test,files_test,sort=True)

    

        
