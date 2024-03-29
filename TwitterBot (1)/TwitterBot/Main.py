from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import metrics
from keras.applications import VGG19

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint


main = tkinter.Tk()
main.title("Detecting Malicious Twitter Bots Using Machine Learning") #designing main screen
main.geometry("1300x1200")

global filename
global dataset, X, Y, X_train, X_test, y_train, y_test
global graph

words = ['bot','cannabis','tweet me','mishear','follow me','updates','every','gorilla','forget']

def getFrequency(bow):
    count = 0
    for i in range(len(words)):
        if words[i] in bow:
            count = count + bow.get(words[i])
    return count        

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    
def runModule1():
    global dataset
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))    

def runModule2():
    global graph
    global X, Y, X_train, X_test, y_train, y_test
    graph = []
    text.delete('1.0', END)
    train = dataset[['screen_name','status','name','followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
    details = train.values
    text.insert(END,"Possible BOT users\n\n")
    users = []
    for i in range(len(details)):
        screen = details[i,0]
        status = details[i,1]
        name = details[i,2]
        followers = int(details[i,3])
        friends = int(details[i,4])
        listed = int(details[i,5])
        favourite = int(details[i,6])
        status_count = int(details[i,7])
        verified = details[i,8]
        if not verified: #check user not verified
            bow = defaultdict(int) #bag of words
            data = str(screen)+" "+str(name)+" "+str(status)#checking screen name, tweets and name
            data = data.lower().strip("\n").strip()
            data = re.findall(r'\w+', data)
            for j in range(len(data)):
                bow[data[j]] += 1  #adding each word frequency to bag of words
            frequency = getFrequency(bow) #getting frequency of BOTS words            
            if frequency > 0 and listed < 16000 and followers < 200: #if condition true then its bots
                users.append(screen)
    text.insert(END,str(users)+"\n")            
    train_attr = dataset[
        ['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
    train_label = dataset[['bot']]

    X = train_attr
    Y = train_label.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print(X.shape)
    print(X_train.shape)
    logreg = LogisticRegression().fit(X_train, y_train)#logistic regression object
    actual = y_test
    pred = logreg.predict(X_test)
    accuracy = accuracy_score(actual, pred) * 100
    graph.append(accuracy)
    precision = precision_score(actual, pred) * 100
    recall = recall_score(actual, pred) * 100
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    text.insert(END,'\nLogistic Regression Accuracy  : '+str(accuracy)+"\n")
    text.insert(END,'Logistic Regression Precision : '+str(precision)+"\n")
    text.insert(END,'Logistic Regression Recall is : '+str(recall)+"\n")
    text.insert(END,'Logistic Regression Area Under Curve is : '+str(auc))

    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def runModule3():
    text.delete('1.0', END)
    urls = []
    details = dataset.values
    for i in range(len(details)):#checking URLS in tweets
        tweets = details[i,14]
        if 'http' in str(tweets):
            urls.append(1)
        else:
            urls.append(0)

    train_attr = dataset[
        ['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
    train_attr["URLS"] = urls #adding URLS to training dataset
    text.insert(END,str(train_attr))
    train_label = dataset[['bot']]
    X1 = train_attr
    Y1 = np.asarray(train_label)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2)

    logreg = LogisticRegression().fit(X_train1, y_train1) #logistic regression object
    actual = y_test1
    pred = logreg.predict(X_test1)
    accuracy = accuracy_score(actual, pred) * 100
    graph.append(accuracy)
    precision = precision_score(actual, pred) * 100
    recall = recall_score(actual, pred) * 100
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    text.insert(END,'\nLogistic Regression Accuracy  : '+str(accuracy)+"\n")
    text.insert(END,'Logistic Regression Precision : '+str(precision)+"\n")
    text.insert(END,'Logistic Regression Recall is : '+str(recall)+"\n")
    text.insert(END,'Logistic Regression Area Under Curve is : '+str(auc))

    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def runModule4():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X_train = X_train.values
    X_test = X_test.values
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    if os.path.exists("model/vgg19_weights.hdf5") == True:
        vgg_model = load_model("model/vgg19_weights.hdf5")
    else:
        #now training VGG19 on Eyepacs dataset
        vgg = VGG19(include_top=False, weights="imagenet", input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        for layer in vgg.layers:
            layer.trainable = False
        vgg_model = Sequential()
        vgg_model.add(vgg)#adding VGG as transfer learning
        #now adding new layers to VGG for classifying eyepacs dataset
        vgg_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        vgg_model.add(MaxPooling2D(pool_size = (1, 1)))
        vgg_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
        vgg_model.add(MaxPooling2D(pool_size = (1, 1)))
        vgg_model.add(Flatten())
        vgg_model.add(Dense(units = 256, activation = 'relu'))
        vgg_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
        vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) 
        model_check_point = ModelCheckpoint(filepath='model/vgg19_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg_model.fit(X_train, y_train, batch_size=16, epochs=40, shuffle=True, callbacks=[model_check_point], verbose=1, validation_data=(X_test, y_test))
    predict = vgg_model.predict(X_test) #now perform prediction on test data
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test1, predict) * 100
    graph.append(accuracy)
    precision = precision_score(y_test1, predict) * 100
    recall = recall_score(y_test1, predict) * 100
    f1 = f1_score(y_test1, predict)
    auc = roc_auc_score(y_test1, predict)
    text.insert(END,'\nVGG19 Accuracy  : '+str(accuracy)+"\n")
    text.insert(END,'VGG19 Precision : '+str(precision)+"\n")
    text.insert(END,'VGG19 Recall is : '+str(recall)+"\n")
    text.insert(END,'VGG19 Area Under Curve is : '+str(auc))

    fpr, tpr, thresholds = metrics.roc_curve(y_test1, predict)
    auc = metrics.auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def plotGraph():
    global graph
    height = graph
    bars = ('ML Bot Accuracy','ML URL Accuracy','VGG19 BOT & URL Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Algorithm Names")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Graph")
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Detecting Malicious Twitter Bots Using Machine Learning')
title.config(bg='goldenrod2', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Tweets Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

module1Button = Button(main, text="Run Module 1 (Extract Tweets)", command=runModule1, bg='#ffb3fe')
module1Button.place(x=520,y=550)
module1Button.config(font=font1) 

module2Button = Button(main, text="Run Module 2 (Recognize Twitter Bots using ML)", command=runModule2, bg='#ffb3fe')
module2Button.place(x=50,y=600)
module2Button.config(font=font1) 

module3Button = Button(main, text="Run Module 3 (Recognize Malicious URLS using ML)", command=runModule3, bg='#ffb3fe')
module3Button.place(x=520,y=600)
module3Button.config(font=font1)

module3Button = Button(main, text="Run Module 4 (Recognize Twitter Bots using VGG19)", command=runModule4, bg='#ffb3fe')
module3Button.place(x=50,y=650)
module3Button.config(font=font1)

module3Button = Button(main, text="Comparison Graph", command=plotGraph, bg='#ffb3fe')
module3Button.place(x=520,y=650)
module3Button.config(font=font1) 


main.config(bg='SpringGreen2')
main.mainloop()
