from scipy import io
from keras.optimizers import SGD
from keras.models import load_model
import scipy
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import merge, Input, Dense, Activation, Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
import time
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature

random.seed(30)

def GetDefaultParameters():
    p = {}
    p['trainIdx'] = 220
    p['validIdx'] = 300
    p['testIdx'] = 473
    p['num_classes'] = 2
    p['dataPath'] = 'FlowerData'
    p['s'] = 224
    p['testIdx'] = 300
    p['validPercent'] = 0.25
    p['seed'] = 30
    p['batch_size'] = 32
    p['epochs'] = 200
    p['patience'] = 6
    p['modelName'] = 'modified_decay01.hdf5'
    return p


def get_data(data_path, s):
    load = scipy.io.loadmat('FlowerDataLabels')
    labels = load['Labels'][0:1][0]
    labels1 = np.ones((labels.shape[0], ), dtype=int)
    labels1 = labels1-labels
    labels = np.transpose(np.array([labels, labels1]))
    data = []
    for i in range(labels.shape[0]):
        tmp = cv2.imread(data_path + '/' + str(i + 1) + '.jpeg')
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp = cv2.resize(tmp, (s, s))
        data.append(tmp)
    data = np.asarray(data)
    return data, labels


def split_data(testIdx, validPercent, data, labels, seed):
    # split for train and test
    trainData = data[0:testIdx]
    testData = data[testIdx:]
    trainLabels = labels[0:testIdx]
    testLabels = labels[testIdx:]

    # split for train and validation
    trainData, validData, trainLabels, validLabels = train_test_split(trainData, trainLabels,
                                                                      test_size=validPercent,
                                                                      random_state=seed)
    return trainData, validData, testData, trainLabels, validLabels, testLabels


def ploting(hist):
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(len(train_loss))

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.show()


def augmentation(trainData, trainLabels,numOfAug):
    if not os.path.exists('augmentation'):
        os.makedirs('augmentation')
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                 shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
                                 vertical_flip= True,fill_mode='nearest')
    j = 1
    for img in trainData:
        img = img.reshape((1,) + img.shape)
        i=0
        for batch in datagen.flow(img, batch_size=1):
            scipy.misc.imsave('augmentation/' + str(j) + '.jpeg', batch[0])
            i += 1
            j += 1
            if i >= numOfAug:
                break  # otherwise the generator would loop indefinitely
    augmLabels = np.array([])
    for i in trainLabels[:,0]:
        tmp = np.repeat(i,numOfAug)
        augmLabels = np.append(augmLabels,tmp)

    labels1 = np.ones((augmLabels.shape[0],), dtype=int)
    labels1 = labels1 - augmLabels
    augmLabels = np.transpose(np.array([augmLabels, labels1]))
    return augmLabels


def load_augmentation_data(labels):
    data = []
    for i in range(labels.shape[0]):
        tmp = cv2.imread( 'augmentation/' + str(i + 1) + '.jpeg')
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        data.append(tmp)
    data = np.asarray(data)
    return data


def run_model(path, patience, epochs, batch_size, model, trainData, trainLabels, validData, validLabels):
    t = time.time()
    # check point
    checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # early stopping
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    callbacks_list = [checkpoint, earlystopper]
    # fit
    hist = model.fit(trainData, trainLabels, callbacks=callbacks_list,
                     batch_size=batch_size, epochs=epochs,
                     verbose=1, validation_data=(validData, validLabels), shuffle=True)
    print('Training time: %s' % (time.time() - t))
    (loss, accuracy) = model.evaluate(validData, validLabels, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return hist


def precision_recall(y_test, y_score):
    # recall = (np.sum(np.multiply(prediction, groundTruth)))/np.sum(groundTruth)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    average_precision = average_precision_score(y_test, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def show_err(err_type, testLabels, testData, prediction):
    object_GT = testLabels[:, 0]
    if err_type == 1:
        # type 1 error: prediction=0 and ground truth is 1
        not_object = prediction[:, 0] < 0.5
        err1 = np.multiply(object_GT, not_object)
        err1 = err1.astype(bool)
        indices_type = err1.argsort()[-5:][::-1]  # 5 biggest type 1 err indices
        indices_type = indices_type[np.argsort(prediction[indices_type, 0])]  # sorted
    else:
        # type 2: prediction=1 ground truth=0
        object = prediction[:, 0] > 0.5
        not_object_GT = np.ones((object_GT.shape[0],), dtype=int) - object_GT
        err2 = np.multiply(not_object_GT, object)
        indices_type = err2.argsort()[-5:][::-1]
        indices_type = indices_type[np.argsort(prediction[indices_type, 0])[::-1][:5]]

    counter = 1
    for idx in indices_type:
        img = testData[idx]
        if err_type == 1:
            grade = "%.3f" % (1 - prediction[idx, 0])
        else:
            grade = "%.3f" % prediction[idx, 0]

        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        # add section at top
        violet = np.zeros((100, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)
        # concat the section
        vcat = cv2.vconcat((violet, constant))
        # add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        sub_title = 'Index: ' + str(counter) + ' score: ' + grade
        counter += 1
        title = 'Err type: ' + str(err_type)
        cv2.putText(vcat, sub_title, (5, 50), font, 0.5, (0, 0, 0), 1)
        cv2.putText(vcat, title, (5, 20), font, 0.5, (0, 0, 0), 1)
        cv2.imshow('Text', vcat)
        cv2.waitKey(0)


def print_res(model, testData, testLabels):
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


params = GetDefaultParameters()
data, labels = get_data(params['dataPath'], params['s'])
trainData, validData, testData, trainLabels, validLabels, testLabels = \
    split_data(params['testIdx'], params['validPercent'], data, labels, params['seed'])
image_input = Input(shape=(params['s'], params['s'], 3))
model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

#######################################################################################
#################################
# modified  net
#################################
# last_layer = model.get_layer('block5_pool').output
# x = Flatten(name='flatten')(last_layer)
# x = Dense(128, activation='relu', name='fc1')(x)
# x = Dense(128, activation='relu', name='fc2')(x)
# out = Dense(params['num_classes'], activation='softmax', name='output')(x)
# custom_vgg_model = Model(image_input, out)
# custom_vgg_model.summary()
# for layer in custom_vgg_model.layers[:-3]:
#     layer.trainable = False
#
# custom_vgg_model.summary()
# opt = SGD(lr=0.001)
# custom_vgg_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
#                  custom_vgg_model, trainData, trainLabels, validData, validLabels)
# ploting(hist)
# model = load_model(params['modelName'])
# print_res(model, testData, testLabels)
# precision_recall(testLabels, model.predict(testData))



##########################################################################################
####################
# # regular pipe
####################
# last_layer = model.get_layer('fc2').output
# out = Dense(params['num_classes'], activation='softmax', name='output')(last_layer)
#
# custom_vgg_model = Model(image_input, out)
# for layer in custom_vgg_model.layers[:-1]:
#     layer.trainable = False
# custom_vgg_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
#                  custom_vgg_model, trainData, trainLabels, validData, validLabels)
#
# ploting(hist)
# model = load_model(params['modelName'])
# print_res(model, testData, testLabels)
# precision_recall(testLabels, model.predict(testData))

################################################################################
####################
# regular pipe with data augmentation
####################
# trainLabels = augmentation(trainData, trainLabels,3)
# trainData = load_augmentation_data(trainLabels)
#
# last_layer = model.get_layer('fc2').output
# out = Dense(params['num_classes'], activation='softmax', name='output')(last_layer)
#
# custom_vgg_model = Model(image_input, out)
# for layer in custom_vgg_model.layers[:-1]:
#     layer.trainable = False
# custom_vgg_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
#                  custom_vgg_model, trainData, trainLabels, validData, validLabels)
#
# ploting(hist)
# model = load_model(params['modelName'])
# print_res(model, testData, testLabels)
# precision_recall(testLabels, model.predict(testData))

######################################################################################################
# 2 augmentation and modified net
##################################
#
# last_layer = model.get_layer('block5_pool').output
# x = Flatten(name='flatten')(last_layer)
# x = Dense(128, activation='relu', name='fc1')(x)
# x = Dense(128, activation='relu', name='fc2')(x)
# out = Dense(params['num_classes'], activation='softmax', name='output')(x)
#
# custom_vgg_model = Model(image_input, out)
# custom_vgg_model.summary()
#
# for layer in custom_vgg_model.layers[:-3]:
#     layer.trainable = False
#
# custom_vgg_model.summary()
# opt = SGD(lr=0.001)
# custom_vgg_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#
# trainLabels = augmentation(trainData, trainLabels,2)
# trainData = load_augmentation_data(trainLabels)
#
# hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
#                  custom_vgg_model, trainData, trainLabels, validData, validLabels)
#
# ploting(hist)
# model = load_model(params['modelName'])
# print_res(model, testData, testLabels)
# precision_recall(testLabels, model.predict(testData))



#####################################################################################################
# train 2 models with augmentation:1 train layers, 2 trained layres
################################################
#
# last_layer = model.get_layer('fc2').output
# opt = SGD(lr=0.001)
# trainLabels = augmentation(trainData, trainLabels, 4)
# trainData = load_augmentation_data(trainLabels)
# for i in range(1, 3):
#     out = Dense(params['num_classes'], activation='softmax', name='output')(last_layer)
#     custom_vgg_model = Model(image_input, out)
#     for layer in custom_vgg_model.layers:
#         layer.trainable = True
#     for layer in custom_vgg_model.layers[:-i]:
#         layer.trainable = False
#     custom_vgg_model.summary()
#     custom_vgg_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
#                  custom_vgg_model, trainData, trainLabels, validData, validLabels)
#
# ploting(hist)
# model = load_model(params['modelName'])
# print_res(model, testData, testLabels)
# precision_recall(testLabels, model.predict(testData))

#####################################################################################
# with dropout & data augmentation=4 &
############################################
# last_layer = model.get_layer('fc2').output
# # Store the fully connected layers
# fc1 = model.layers[-3]
# fc2 = model.layers[-2]
# predictions = model.layers[-1]
#
# # Create the dropout layers
# dropout1 = Dropout(0.5)
# dropout2 = Dropout(0.5)
#
# # Reconnect the layers
# x = dropout1(fc1.output)
# x = fc2(x)
# x = dropout2(x)
# predictors = predictions(x)
#
# opt = SGD(lr=0.001)
# trainLabels = augmentation(trainData, trainLabels, 4)
# trainData = load_augmentation_data(trainLabels)
#
# out = Dense(params['num_classes'], activation='softmax', name='output')(predictors)
# custom_vgg_model = Model(image_input, out)
#
# for layer in custom_vgg_model.layers[:-2]:
#     layer.trainable = False
#
# custom_vgg_model.summary()
# custom_vgg_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
#                  custom_vgg_model, trainData, trainLabels, validData, validLabels)
#
# ploting(hist)
# model = load_model(params['modelName'])
# print_res(model, testData, testLabels)
# precision_recall(testLabels, model.predict(testData))
###############################################################################3
# train two layers
#####################################################3
last_layer = model.get_layer('fc2').output
opt = SGD(lr=0.001, decay=0.01)

out = Dense(params['num_classes'], activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)

for layer in custom_vgg_model.layers[:-2]:
    layer.trainable = False
custom_vgg_model.summary()
custom_vgg_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

hist = run_model(params['modelName'], params['patience'], params['epochs'], params['batch_size'],
                 custom_vgg_model, trainData, trainLabels, validData, validLabels)

ploting(hist)
model = load_model(params['modelName'])
print_res(model, testData, testLabels)
precision_recall(testLabels[:, 0], model.predict(testData)[:, 0])
# showing 5 biggest err for each err type
# show_err(1,testLabels,testData,model.predict(testData))
# show_err(2,testLabels,testData,model.predict(testData))