
from datetime import datetime as dt
import csv, os, glob
from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
# from sklearn.preprocessing import label_binarize
from scipy import interp

import tensorflow.keras as k
from tensorflow.math import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt


def draw_CM(label, predicted):
    cm = confusion_matrix(label, predicted)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # true : false rate
    true = 0
    false = 0
    for i, j in enumerate(label):
        if j != predicted[i]:
            false += 1
        else: true += 1

    classification_report = metrics.classification_report(label, predicted)
    multilabel_to_binary_matrics = metrics.multilabel_confusion_matrix(label, predicted)

    return plt.show(), print('true rate: ', true), print('false rate: ', false), print(), print('='*10, 'classification_report: ', '\n', classification_report), print('='*10, 'multilabel_to_binary_matrics by class_num: ','\n','[[TN / FP] [FN / TP]]','\n', multilabel_to_binary_matrics)
    
def draw_ROC_AUC(x, y, category_names):
    n_classes = len(category_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], x[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', 
            color='deeppink', linestyle=':', linewidth=1)

    plt.plot(fpr["macro"], tpr["macro"],
            label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
            color='navy', linestyle=':', linewidth=1)

    colors = (['purple', 'pink', 'red', 'green', 'yellow', 'cyan', 'magenta', 'blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1, label=f'Class {i} ROC curve (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC & AUC')
    plt.legend(loc="lower right")

    return plt.show()

path = '/home/z/data/label/'
for file in glob.glob(path + "*.csv"):
    df = pd.read_csv(file)
    df = df[['device_field03', 'illuminance_onoff', 'device_data_reg_dtm']]
    light_data = df['device_field03']
    label_data = df['illuminance_onoff']
    time_data = df['device_data_reg_dtm']
    start = 230
    end = 245


    # 9 to 0
    classes = ["idle", "on", "off"]
    class_list = list(range(len(classes)))
    prev = 0
    idle = 0
    on = 1
    off = 2
    for row, label in label_data.iteritems():
        if prev == 0 and label == 1:
            df.at[row, 'illuminance_onoff'] = on
        elif prev == 1 and label == 0:
            df.at[row, 'illuminance_onoff'] = off
        else:
            df.at[row, 'illuminance_onoff'] = idle
        prev = label


    # Nomalize
    mx = light_data.max()
    norm_data = light_data / mx


    # Split
    split_index = int(len(light_data)*0.9)
    train_data, test_data = norm_data[:split_index], norm_data[split_index:]
    train_label, test_label = label_data[:split_index], label_data[split_index:]

    test_data.reset_index(drop=True, inplace=True)
    test_label.reset_index(drop=True, inplace=True)

    def to_sequences(seq_size, t1, t2):
        x = []
        y = []
        for i in range(len(t1)-seq_size):
            ta1 = t1[i:(i+seq_size)]
            ta2 = t2[i-1+seq_size]
            # ta1 = [[x] for x in ta1]
            x.append(ta1)
            y.append(ta2)

        return np.array(x), np.array(y)

    timesteps = 3
    x_train, y_train = to_sequences(timesteps,train_data, train_label)
    x_test, y_test = to_sequences(timesteps,test_data, test_label)


    # one-hot encoding
    def label_maker(target):
        result = []
        for i in target:
            cal = [0,0,0]
            cal[i] = 1
            result.append(cal)

        return np.array(result)

    y_train = label_maker(y_train)
    y_test = label_maker(y_test)


    # model
    input = k.Input(shape=(timesteps, ))
    # input = k.Input(shape=(1, timesteps))

    # x = k.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid', dropout=0, recurrent_dropout=0, unroll=False, use_bias=True)(input)
    x = k.layers.Dense(128, activation="sigmoid")(input)
    # x = k.layers.Dense(64, activation="sigmoid")(x)
    # x = k.layers.Dense(32, activation="sigmoid")(x)
    # x = k.layers.Dense(16, activation="sigmoid")(x)
    x = k.layers.Dense(8, activation="sigmoid")(x)

    output = k.layers.Dense(3, activation="sigmoid")(x)

    model = k.Model(input, output)

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # fit
    log_path = "logs/" + dt.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    es = EarlyStopping(monitor="val_loss", patience=3, mode="auto", verbose=2)

    history = model.fit(x_train, y_train, validation_split=0.1, batch_size=12, epochs=100, verbose=1, callbacks=[es]) # callbacks=[es, tensorboard_callback])
    print(history)
    # plot
    pd.DataFrame(history.history).plot(figsize=(16,10), grid=1, xlabel="epoch", ylabel="accuracy")
    plt.show()


# save model
file_name =  "model/light_detector_" + dt.now().strftime("%Y%m%d-%H%M%S")
model_format = ".h5"
model_name = file_name + model_format
model.save(model_name)


# evauate
model = k.models.load_model(model_name)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f'test_loss: {loss} test_accuracy: {acc}')


predict = model.predict(x_test)
predicted = np.argmax(predict, axis=1)
y1 = np.argmax(y_test, axis=1)


# CM
draw_CM(y1, predicted)

# ROC, AUC
draw_ROC_AUC(predict, y_test, classes)

