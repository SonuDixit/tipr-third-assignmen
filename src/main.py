import pickle
import argparse
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from utils import *
from cnn import CNN
from cnn_cifar import CNN as CNN2

def print_res(test_label,pred_lab):
    acc = accuracy_score(test_label, pred_lab)
    macro = f1_score(test_label, pred_lab, average="macro")
    micro = f1_score(test_label, pred_lab, average="micro")
    print("Test Accuracy :: " + str(acc * 100))
    print("Test Macro F1-score :: " + str(macro * 100))
    print("Test Micro F1-score :: " + str(micro * 100))

parser = argparse.ArgumentParser()
parser.add_argument('-test_file', '--test-data', help='path of test file', required=False)
parser.add_argument('-train_path', '--train-data', help='path of train directory', required=False)
parser.add_argument('-dataset', '--dataset', help='data file name Dolphins,PubMed, Twitter', required=False)
parser.add_argument('-config', '--filter-configuration',type=str, nargs='+', help='list of ints', required=False)

args = vars(parser.parse_args())
hidden_nodes =[]
if not args["filter_configuration"] is None:
    for i in range(len(args["filter_configuration"])):
        if i==0:
            hidden_nodes.append(int(args["filter_configuration"][0][1:]))
        elif i == (len(args["filter_configuration"])-1):
            hidden_nodes.append(int(args["filter_configuration"][i][:-1]))
        else:
            hidden_nodes.append(int(args["filter_configuration"][i]))
# print(args.keys())
"""
test-data has been converted to test_data
test-label has been converted to test_label
"""
if args["dataset"] == "Fashion-MNIST" :
    if not args["train_data"] is None:
        tr_data, tr_label = load_mnist(args["train_data"])
        tr_data = tr_data.reshape(tr_data.shape[0],28,28,1)
        net_4 = CNN(input_dim=(28,28,1),
                 filter_size=hidden_nodes,
                 output_dim=10,
                 activation=["relu"] * len(hidden_nodes),
                 filters=[32] * len(hidden_nodes),
                 embedding_dim = 32,
                 dropout = 0.2)
        net_4.model.summary()
        net_4.fit(tr_data, tr_label)
        test_data, test_label = load_mnist(args["test_data"])
        test_data = test_data.reshape(test_data.shape[0],28,28,1)
        pred_lab = net_4.predict(test_data)
        print_res(test_label,pred_lab)
    else:
        test_data,test_label = load_mnist(args["test_data"])
        test_data = test_data.reshape(test_data.shape[0],28,28,1)
        net4 = tf.keras.models.load_model("mnist_best.h5")

        pred_lab = np.argmax(net4.predict(test_data),axis=1)
        print_res(test_label, pred_lab)
elif args["dataset"] in ["Cifar-10", "Cifar 10"]:
    if not args["train_data"] is None:
        images, label = get_batch_data_cifar(1)
        for i in range(2, 6):
            i, lab = get_batch_data_cifar(i)
            images = np.vstack((images, i))
            label.extend(lab)
        mean = np.mean(images, axis=(0, 1, 2, 3))
        std = np.std(images, axis=(0, 1, 2, 3))
        images = (images - mean) / (std + 1e-7)
        images = images.reshape(images.shape[0], 32, 32, 3)

        net_4 = CNN2(input_dim=(32,32,3),
                 filter_size=hidden_nodes,
                 output_dim=10,
                 activation=["relu"]*len(hidden_nodes),
                 filters=[32]* len(hidden_nodes),
                 embedding_dim = 128,
                 dropout = 0.2)
        net_4.fit(images, label)
        
        test_data, test_label = get_batch_data_cifar_test()
        test_data = test_data.reshape(test_data.shape[0],32,32,3)
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
    else:
        test_data,test_label = get_batch_data_cifar_test()
        ## load a NN
        net4 = tf.keras.models.load_model("cifar_new.h5")
        test_data = test_data.reshape(test_data.shape[0],32,32,3)
        pred_lab = np.argmax(net4.predict(test_data),axis=1)
        print_res(test_label, pred_lab)