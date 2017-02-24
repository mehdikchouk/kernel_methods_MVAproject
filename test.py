import numpy as np
from multiclass_classification import *


Xtr = np.genfromtxt ('Xtr.csv', delimiter=",")
Ytr = np.genfromtxt ('Ytr.csv', delimiter=",",skip_header=1)
Ytr=Ytr[:,1]
Xtr=Xtr[:,:-1]

# from extract_features import *
# X = np.genfromtxt('Xtr.csv', delimiter=',')
# X = np.delete(X, np.s_[-1:], 1)
# n_img = X.shape[0]
# X = X.reshape(n_img, 3, 32, 32).transpose(0,2,3,1)
#
# # Use the first image to determine feature dimensions
# feat, dim = extract_hog(X[0],4)
# print "dim : ", dim
# X_hog = np.zeros((n_img, dim))
#
# # Extract features for the rest of the images
# for i in range(n_img):
#     feat, dim = extract_hog(X[i],4)
#     X_hog[i, :] = feat
#
# Xtr=X_hog


#save the trained model
'''
np.save('parameters_linearSVM_fulltrain.npy', parameters)
# Load the model
parameters = np.load('parameters_linearSVM_fulltrain.npy').item()
'''
#split the data and work on a test set for evaluation
# Ytr=np.array([Ytr])
# data=np.concatenate((Xtr,Ytr.T),axis=1)
# msk = np.random.rand(len(data)) < 0.8
# train = data[msk]
# test = data[~msk]
# Xtrain=train[:,:-1]
# Ytrain=train[:,-1]
# Xtest=test[:,:-1]
# Ytest=test[:,-1]
#
#
# parameters=one_vs_all(Xtrain,Ytrain,kernel=linear_kernel,C=1)
# number_of_classes=10
# y_hat=predict_multiclass(Xtest,number_of_classes,parameters)
# correct = np.sum(y_hat == Ytest)
# acc=correct/float(len(y_hat))
# print("%d out of %d predictions correct" % (correct, len(y_hat)))
# print "Accuracy on test : ",acc
# y_hat_train=predict_multiclass(Xtrain,number_of_classes,parameters)
# correct = np.sum(y_hat_train == Ytrain)
# acc=correct/float(len(y_hat_train))
# print("%d out of %d predictions correct" % (correct, len(y_hat)))
# print "Accuracy on train : ",acc
#
#
#
# #Tune the hyperparameters
# C_values=[0.0001,0.1,1,10,10000]
# accuracy_list=[]
# for i in C_values:
#     parameters=one_vs_all(Xtrain,Ytrain,kernel=linear_kernel,C=i)
#     number_of_classes=10
#     y_hat=predict_multiclass(Xtest,number_of_classes,parameters)
#     correct = np.sum(y_hat == Ytest)
#     acc=correct/float(len(y_hat))
#     print("%d out of %d predictions correct" % (correct, len(y_hat)))
#     print "Accuracy : ",acc
#     accuracy_list.append(acc)
#

#Testing HOG features
from histogram import hog


train=np.zeros((Xtr.shape[0],1728))
for i in range(Xtr.shape[0]):
    im1=Xtr[i,0:1024].reshape((32,32))
    im2=Xtr[i,1024:2048].reshape((32,32))
    im3=Xtr[i,2048:3072].reshape((32,32))
    hog_features1=hog(im1,flatten=True)
    hog_features2=hog(im2,flatten=True)
    hog_features3=hog(im3,flatten=True)
    vect=np.concatenate((hog_features1,hog_features2,hog_features3))
    train[i,:]=vect

#split the data and work on a test set for evaluation
Ytr=np.array([Ytr])
data=np.concatenate((train,Ytr.T),axis=1)
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]
Xtrain=train[:,:-1]
Ytrain=train[:,-1]
Xtest=test[:,:-1]
Ytest=test[:,-1]

#save the split to keep the same structure
# np.savetxt("Xtrain.csv", Xtrain, delimiter=",")
# np.savetxt("Ytrain.csv", Ytrain, delimiter=",")
# np.savetxt("Xtest.csv", Xtest, delimiter=",")
# np.savetxt("Ytest.csv", Ytest, delimiter=",")

import numpy as np
from multiclass_classification import *
Xtrain = np.genfromtxt ('Xtrain.csv', delimiter=",")
Ytrain = np.genfromtxt ('Ytrain.csv', delimiter=",")
Xtest = np.genfromtxt ('Xtest.csv', delimiter=",")
Ytest = np.genfromtxt ('Ytest.csv', delimiter=",")




parameters=one_vs_all(Xtrain,Ytrain,kernel=histogram_intersection_kernel,C=100)
np.save('parameters_HIKSVM_Xtrain.npy', parameters)
parameters = np.load('parameters_HIKSVM_Xtrain.npy').item()



number_of_classes=10
y_hat=predict_multiclass(Xtest,number_of_classes,parameters)
correct = np.sum(y_hat == Ytest)
acc=correct/float(len(y_hat))
print("%d out of %d predictions correct" % (correct, len(y_hat)))
print "Accuracy on test : ",acc
y_hat_train=predict_multiclass(Xtrain,number_of_classes,parameters)
correct = np.sum(y_hat_train == Ytrain)
acc=correct/float(len(y_hat_train))
print("%d out of %d predictions correct" % (correct, len(y_hat_train)))
print "Accuracy on train : ",acc


''' write out the results to a csv file'''
import csv
import sys

f = open('Yte.csv', 'wt')
try:
    writer = csv.writer(f)
    writer.writerow( ('Id','Prediction') )
    for i in range(Xte.shape[0]):
        writer.writerow( (i+1, Yte[i]) )
finally:
    f.close()
