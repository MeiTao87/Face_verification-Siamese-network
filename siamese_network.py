import os
import cv2
import numpy as np
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
from keras.models import Model
import random
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import pickle

def calculate_dist(vectors):
    x, y = vectors
    # res = square(x - y)
    # res = max (res, 1e-7)
    # sqrt(res)
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def calculate_dist_output_shape(vectors): 
    shape_1 = vectors[0].shape # 128, 1
    shape_2 = vectors[1].shape # 128, 1
    return (shape_1[0], 1)

# def contrastive_loss(y_true, y_pred):
#     margin = 1
#     return K.mean(y_true * K.square(y_pred) +
#                   (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# from https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss


class Siamese_network():
    def __init__(self):
        self.labels = None
        self.threshold = 0.5
        self.vgg_clude_top = False
        self.vgg16_model = None
        self.input_shape = None
        self.weight_file_path = './models/model_weight.h5'
        self.config_file_path = './models/model_config.npy'
        self.database_path = './data/database.pkl'
        self.database = None
        self.img_path = '/content/drive/MyDrive/face_data'
        
    def create_database(self):
        names = os.listdir(self.img_path)
        self.database = dict()
        ########################################
        # change this 20, it is used for debug #
        ########################################
        for i in range(20): 
            name = names[i]
            whole_path = os.path.join(self.img_path, name)
            if len(os.listdir(whole_path)) >= 2:
                self.database[name] = []
                print(f'i: {i}, name is {name}')
                for img in os.listdir(whole_path):
                    # if img.endswith('face.jpg'):
                    _img_path = os.path.join(self.img_path, name, img)
                    name_encoding = self.img_to_encoding(_img_path)
                    # name_encoding = img_path
                    self.database[name].append(name_encoding)
        try:
            os.mkdir('./data')
        except:
            pass
        with open(self.database_path, "wb") as pkl_handle:
	          pickle.dump(self.database, pkl_handle)

    def load_database(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as pkl_handle:
	            self.database = pickle.load(pkl_handle)
        else:
            self.create_database()

    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    def metrics(self, y_true, y_pred):
        ''' Compute classification accuracy with a fixed threshold on distances.
        K.mean(K.equal(y_true, K.cast(y_pred < self.threshold, y_true.dtype))) '''
        y_pred_thres = K.cast(y_pred<self.threshold, y_true.dtype)
        return K.mean(K.equal(y_true, y_pred_thres))
        
    def create_network(self, input_shape):
        base_model = self.create_base_network(input_shape)
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        base_output_a = base_model(input_a)
        base_output_b = base_model(input_b)
        dist = Lambda(calculate_dist, 
                    output_shape=calculate_dist_output_shape)([base_output_a, base_output_b])
        model = Model([input_a, input_b], dist)
        optimizer = Adam()
        model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[self.metrics])
        print(model.summary())
        return model

    def create_vgg16_model(self):
        model = VGG16(include_top=self.vgg_clude_top)
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def img_to_encoding(self, img_path):
        if self.vgg16_model is None:
            self.vgg16_model = self.create_vgg16_model()
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        return self.vgg16_model.predict(input)

    # database is a dict, names is a list containing all the keys in the database
    def create_pairs(self, names):
        number_classes = len(self.database)
        pairs = []
        labels = []
        for j in range(len(names)): # name is a list 
            name = names[j]
            copy_names = names.copy()
            del copy_names[j]
            imgs_list = self.database[name]
            for i in range(len(imgs_list)):
                img_a = imgs_list[i] # choose anchor
                # delete the ith element from imgs_list
                copy_img_list = imgs_list.copy()
                del copy_img_list[i]
                img_p =  random.choice(copy_img_list)# choose positive
                pairs += [[img_a, img_p]] # label 1
                
                # negative image
                # first choose one from copy_names
                n_class_index = np.random.randint(0, len(copy_names))
                # then choose random one from this negative class
                # which class?
                n_class = self.database[copy_names[n_class_index]]
                img_n = random.choice(n_class)
                pairs += [[img_a, img_n]] # label 0
                labels += [1, 0]
                pairs, labels = np.array(pairs), np.array(labels)
        return pairs, labels
    
    # database{'mt': [encoding of mt's pictures],
    #           'zwq': [encoding of zwq's pictures]}
    # names ['mt', 'zwq', ...]
    def fit(self, epochs=None, batch_size=None):
        
        self.load_database()
        if batch_size is None:
            batch_size = 16
        if epochs is None:
            epochs = 20
        for name, encoding in self.database.items():
            self.input_shape = encoding[0].shape
            break
        self.vgg16_model = self.create_vgg16_model()
        self.model = self.create_network(input_shape=self.input_shape)
        architecture_file_path = './models/model_architecture.h5'
        with open(architecture_file_path, 'w') as f:
            f.write(self.model.to_json())
        # open(architecture_file_path, 'w').write(self.model.to_json())
        
        names = []
        self.labels = dict()
        for name in self.database.keys():
            names.append(name)
            self.labels[name] = len(self.labels)

        self.config = dict()
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold
        self.config['vgg16_include_top'] = self.vgg16_include_top
        np.save(self.config_file_path, self.config)

        training_pairs, labels = self.create_pairs(names)
        print(f'data set pairs: {training_pairs.shape}')
        checkpoint = ModelCheckpoint(self.weight_file_path)
        self.model.fit([training_pairs[:,0], training_pairs[:,1]], labels,
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[checkpoint])
        
        self.model.save_weights(self.weight_file_path)
    
    def load_model(self):
        self.config = np.load(self.config_file_path, allow_pickle=True).item()
        self.labels = self.config['labels']
        self.input_shape = self.config['input_shape']
        self.threshold = self.config['threshold']
        self.vgg16_include_top = self.config['vgg16_include_top']
        self.load_database() 
        self.vgg16_model = self.create_vgg16_model()
        self.model = self.create_network(input_shape=self.input_shape)
        self.model.load_weights(self.weight_file_path)

    def verify(self, img_path, person, threshold=None):
        """
        person -- string of the person's name
        """
        if threshold is not None:
            self.threshold = threshold
        # turn images into encodings
        encoding = self.img_to_encoding(img_path)
        # make pairs
        input_pair = []
        x = self.database[person]
        for i in range(len(x)):
            input_pair.append([encoding, x[i]])
        input_pair = np.array(input_pair)
        dist = self.model.predict([input_pair[:,0], input_pair[:,1]])
        aver_dist = np.average(dist, axis=-1)[0]
        # decide whether it's the same person
        if aver_dist < self.threshold:
            print(f'It\'s {str(person)}, distance is {aver_dist}.')
            is_same = True
        else:
            print(f'It\'s not {str(person)}, distance is {aver_dist}.')
            is_same = False
        return aver_dist, is_same

    def compare(self, img1_path, img2_path, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        encoding1 = self.img_to_encoding(img1_path)
        encoding2 = self.img_to_encoding(img2_path)
        input_pair = [[encoding1, encoding2]]
        input_pair = np.array(input_pair)
        dist = self.model.predict([input_pair[:,0], input_pair[:,1]])[0][0]
        if dist < self.threshold:
            print(f'It\'s the same person, distance is {dist}.')
            is_same = True
        else:
            print(f'It\'s not the same person, distance is {dist}.')
            is_same = False
        return dist, is_same

    def who_is_this(self, img_path, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        encoding = self.img_to_encoding(img_path)
        # iterate through all names
        possible_person = None
        min_dist = 100
        for name in self.database.keys():
            x = self.database[name]
            input_pair = []
            for i in x:
                input_pair.append([encoding, i])
            input_pair = np.array(input_pair)
            dist = self.model.predict([input_pair[:,0], input_pair[:,1]])
            aver_dist = np.average(dist, axis=-1)[0]
            if aver_dist < min_dist:
                min_dist = aver_dist
                possible_person = name
        if min_dist <= self.threshold:
            print(min_dist)
            print(f'This person might be {possible_person}.')
        else:
            print(min_dist)
            print(f'Did not find this person in database.')


def main():
    try:
        os.mkdir('./models')
    except:
        pass
    # test of training
    fnet = Siamese_network()
    fnet.vgg16_include_top = False
   
    # fnet.fit(epochs=5)
    fnet.load_model()
    
    fnet.verify(img_path='/content/drive/MyDrive/face_data/Silvio_Fernandez/Silvio_Fernandez_0001.jpg', person='Silvio_Fernandez')
    fnet.verify(img_path='/content/drive/MyDrive/face_data/Silvio_Fernandez/Silvio_Fernandez_0001.jpg', person='Greg_Gilbert')
    fnet.verify(img_path='/content/drive/MyDrive/face_data/Silvio_Fernandez/Silvio_Fernandez_0001.jpg', person='Larry_Brown')
    fnet.verify(img_path='/content/drive/MyDrive/face_data/Silvio_Fernandez/Silvio_Fernandez_0001.jpg', person='Bridget_Fonda')
main()