import numpy as np
from numpy import genfromtxt
import csv
import torch
from torch.utils.data import DataLoader, sampler, Dataset
import torchvision.transforms as T

def ReadData(data_file, label_file):
    # read in data from csv
    my_data = genfromtxt(data_file, delimiter=',')
    
    labels = []
    # read in labels
    with open(label_file, 'rt') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            labels.append(row[0])
        
    labels = np.asarray(labels)
    
    n, d = np.shape(my_data)
    my_data = my_data[0:,:(d-1)]
    
    # change labels from strings to integers
    integer_labels = np.zeros(np.shape(labels)[0])
    for i in range(len(labels)):
        label_i = labels[i]
        if label_i == 'asphalt':
            integer_labels[i] = 0
        elif label_i == 'cork':
            integer_labels[i] = 1
        elif label_i == 'grass':
            integer_labels[i] = 2
        elif label_i == 'gravel':
            integer_labels[i] = 3
        elif label_i == 'lab':
            integer_labels[i] = 4
        elif label_i == 'laminate_wood':
            integer_labels[i] = 5
        elif label_i == 'pebble':
            integer_labels[i] = 6
        elif label_i == 'sand':
            integer_labels[i] = 7
    labels = integer_labels
    
    
    # try once again to shuffle the data
    shuffler = np.random.permutation(len(labels))
    my_data = my_data[shuffler]
    labels = labels[shuffler]
 
    return my_data, labels

class TactileDataset(Dataset):
    """SAIL-R Tactile dataset."""

    def __init__(self, x, y, transform):
        """
        Args:
            x (numpy array): the data, where each row is a sample
            y (numpy array): train labels
            transform (callable, optional): Optional transform to be applied
                on a sample. 
        """
        self.x = x
        self.y = y
        
        self.transform = transform
        self.tensorTransform = T.Compose([T.ToTensor()])
        
        self.HF = [0]
        self.LF = [4,5]
        self.D = [1,2,3,6]
        self.G = [7]

    def __len__(self):
        return np.shape(self.y)[0]

    def __getitem__(self, idx):
        
        x = self.x[idx,0:414]
        f = self.x[idx,414:447]            # ********** f is the feature vector **********
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.from_numpy(x)
        
        if int(y) in self.HF:
            z = 0
        elif int(y) in self.LF:
            z = 1
        elif int(y) in self.D:
            z = 2
        elif int(y) in self.G:
            z = 3

         # ********** f is stored in 'custom_features' **********
        sample = {'tactile_data': x, 'custom_features': f, 'terrain': y, 'class': z}

        return sample
    
def ReshapeTransform(x):
    return x.reshape(6,1,69)

def NormalizeData(train_data, eval_data, test_data):
    n_train,_ = np.shape(train_data)
    n_eval,_ = np.shape(eval_data)
    n_test,_ = np.shape(test_data)
    
    x_train = train_data[:,0:414].reshape([n_train,6,69])
    x_eval = eval_data[:,0:414].reshape([n_eval,6,69])
    x_test = test_data[:,0:414].reshape([n_test,6,69])
    
    # normalize those channels first
    # calculate mean and std of training data for each channel
    for i in range(6):
        channel_i = np.asarray(x_train[:,i,:])
        channel_mean = np.mean(channel_i)
        channel_std = np.std(channel_i)

        # normalize the data
        x_train[:,i,:] = (x_train[:,i,:] - channel_mean)/channel_std
        x_eval[:,i,:] = (x_eval[:,i,:] - channel_mean)/channel_std
        x_test[:,i,:] = (x_test[:,i,:] - channel_mean)/channel_std
        
    # reshape back
    train_data[:,0:414] = x_train.reshape([n_train, 414])
    eval_data[:,0:414] = x_eval.reshape([n_eval, 414])
    test_data[:,0:414] = x_test.reshape([n_test, 414])
    
    
    # now we normalize the features
    for i in range(33):
        f_i = train_data[:,(414+i)]
        f_i_mean = np.mean(f_i)
        f_i_std = np.std(f_i)
        if f_i_std == 0.0:
            print(i)
            train_data[:,(414+i)] = (train_data[:,(414+i)] - f_i_mean)
            test_data[:,(414+i)] = (test_data[:,(414+i)] - f_i_mean)
            eval_data[:,(414+i)] = (eval_data[:,(414+i)] - f_i_mean)
        else:
            train_data[:,(414+i)] = (train_data[:,(414+i)] - f_i_mean)/f_i_std
            test_data[:,(414+i)] = (test_data[:,(414+i)] - f_i_mean)/f_i_std
            eval_data[:,(414+i)] = (eval_data[:,(414+i)] - f_i_mean)/f_i_std
        
    return train_data, eval_data, test_data

def SeparateData(data, labels, normalize):
    n, d = np.shape(data)
    num_train = 4189
    num_eval = 896
    num_test = 896
    num_each_terrain = 112
    
    train_data = []
    eval_data = []
    test_data = []

    train_labels = []
    eval_labels = []
    test_labels = []

    eval_dict = {}
    test_dict = {}
    
    for i in range(n):
        data_i = data[i,:]
        label_i = labels[i]
        
        if label_i not in eval_dict:
            eval_dict[label_i] = 1
            eval_data.append(data_i)
            eval_labels.append(label_i)
        elif eval_dict[label_i] < num_each_terrain:
            eval_dict[label_i] += 1
            eval_data.append(data_i)
            eval_labels.append(label_i)
        elif label_i not in test_dict:
            test_dict[label_i] = 1
            test_data.append(data_i)
            test_labels.append(label_i)
        elif test_dict[label_i] < num_each_terrain:
            test_dict[label_i] += 1
            test_data.append(data_i)
            test_labels.append(label_i)
        else:
            train_data.append(data_i)
            train_labels.append(label_i)
        
    train_data = np.asarray(train_data)
    eval_data = np.asarray(eval_data)
    test_data = np.asarray(test_data)

    train_labels = np.asarray(train_labels)
    eval_labels = np.asarray(eval_labels)
    test_labels = np.asarray(test_labels)
    
    if normalize:
        train_data, eval_data, test_data = NormalizeData(train_data, eval_data, test_data)
    
    return train_data, train_labels, eval_data, eval_labels, test_data, test_labels



def CreateDataloaders(train_data, train_labels, eval_data, eval_labels, test_data, test_labels, bs):
    trainDataset = TactileDataset(train_data, train_labels, ReshapeTransform)
    evalDataset = TactileDataset(eval_data, eval_labels, ReshapeTransform)
    testDataset = TactileDataset(test_data, test_labels, ReshapeTransform)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=bs, shuffle=True)
    evalLoader = torch.utils.data.DataLoader(evalDataset, batch_size=bs, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=bs, shuffle=True)
    
    return trainLoader, evalLoader, testLoader

def GetProcessedData(data_file, label_file, bs=1, normalize=True):
    data, labels = ReadData(data_file, label_file)
    train_data, train_labels, eval_data, eval_labels, test_data, test_labels = SeparateData(data, labels, normalize)
    return CreateDataloaders(train_data, train_labels, eval_data, eval_labels, test_data, test_labels, bs)
    
    
def Terrain2Class(label):
    HF = [0]
    LF = [4,5]
    D = [1,2,3,6]
    G = [7]
    if int(label) in HF:
        z = 0
    elif int(label) in LF:
        z = 1
    elif int(label) in D:
        z = 2
    elif int(label) in G:
        z = 3
    return z
    