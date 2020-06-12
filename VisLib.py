import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import ProcessData as dp
import torch
import itertools

class_labels = ['HF','LF','D','G']
terrain_labels_old = ['concrete','wood chips', 'grass', 'gravel','laminate wood', 'waxed tile', 'pebble', 'sand']
terrain_labels = ['concrete','waxed tile','laminate wood','wood chips', 'grass', 'gravel', 'pebble', 'sand']

def CreateConfusionMatrix(correct_dict, incorrect_dict):
    print(correct_dict)
    print(incorrect_dict)
    n = len(correct_dict)
    if n > 4:
        correct_dict, incorrect_dict = ReorderTerrain(correct_dict, incorrect_dict)
    confusion_array = np.zeros([n,n])
    totals = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                confusion_array[i,j] = correct_dict[i]
                totals[i] += correct_dict[i]
            elif i in incorrect_dict:
                if j in incorrect_dict[i]:
                    confusion_array[i,j] = incorrect_dict[i][j]
                    totals[i] += incorrect_dict[i][j]
                else:
                    confusion_array[i,j] = 0
            else: 
                confusion_array[i,j] = 0
        confusion_array[i,:] /= totals[i]
    return confusion_array

def ReorderTerrain(correct_dict, incorrect_dict):
    new_correct_dict = {}
    new_incorrect_dict = {}
    int_incorrect_dict = {}
    for i in range(8):
        new_incorrect_dict[i] = {}
        int_incorrect_dict[i] = {}
    new_correct_dict[0] = correct_dict[0]
    new_correct_dict[1] = correct_dict[4]
    new_correct_dict[2] = correct_dict[5]
    new_correct_dict[3] = correct_dict[1]
    new_correct_dict[4] = correct_dict[2]
    new_correct_dict[5] = correct_dict[3]
    new_correct_dict[6] = correct_dict[6]
    new_correct_dict[7] = correct_dict[7]
    
    int_incorrect_dict[0] = incorrect_dict[0]
    int_incorrect_dict[1] = incorrect_dict[4]
    int_incorrect_dict[2] = incorrect_dict[5]
    int_incorrect_dict[3] = incorrect_dict[1]
    int_incorrect_dict[4] = incorrect_dict[2]
    int_incorrect_dict[5] = incorrect_dict[3]
    int_incorrect_dict[6] = incorrect_dict[6]
    int_incorrect_dict[7] = incorrect_dict[7]
    
    for i in range(8):
        if 0 in int_incorrect_dict[i]:
            new_incorrect_dict[i][0] = int_incorrect_dict[i][0]
        if 4 in int_incorrect_dict[i]:
            new_incorrect_dict[i][1] = int_incorrect_dict[i][4]
        if 5 in int_incorrect_dict[i]:
            new_incorrect_dict[i][2] = int_incorrect_dict[i][5]
        if 1 in int_incorrect_dict[i]:
            new_incorrect_dict[i][3] = int_incorrect_dict[i][1]
        if 2 in int_incorrect_dict[i]:
            new_incorrect_dict[i][4] = int_incorrect_dict[i][2]
        if 3 in int_incorrect_dict[i]:
            new_incorrect_dict[i][5] = int_incorrect_dict[i][3]
        if 6 in int_incorrect_dict[i]:
            new_incorrect_dict[i][6] = int_incorrect_dict[i][6]
        if 7 in int_incorrect_dict[i]:
            new_incorrect_dict[i][7] = int_incorrect_dict[i][7]
       

        
    return new_correct_dict, new_incorrect_dict
                    
def ReshapeTerrainCM(CM):
    CM_new = np.zeros([8,8])
    CM_new2 = np.zeros([8,8])
    CM_new[0,:] = CM[0,:]
    CM_new[1,:] = CM[4,:]
    CM_new[2,:] = CM[5,:]
    CM_new[3,:] = CM[1,:]
    CM_new[4,:] = CM[2,:]
    CM_new[5,:] = CM[3,:]
    CM_new[6,:] = CM[6,:]
    CM_new[7,:] = CM[7,:]
    
    for i in range(8):
        CM_new2[i,0] = CM_new[i,0]
        CM_new2[i,1] = CM_new[i,4]
        CM_new2[i,2] = CM_new[i,5]
        CM_new2[i,3] = CM_new[i,1]
        CM_new2[i,4] = CM_new[i,2]
        CM_new2[i,5] = CM_new[i,3]
        CM_new2[i,6] = CM_new[i,6]
        CM_new2[i,7] = CM_new[i,7]

    return CM_new2
    
def PlotConfusionMatrix(confusion_array, labels, title="Confusion Matrix"):
    df_cm = pd.DataFrame(confusion_array, index = labels,
                  columns = labels)
    plt.rcParams.update({'font.size': 14})
    tick_marks = np.arange(len(labels))
    plt.figure(figsize = (10,7))
    plt.yticks(tick_marks, labels)
    g = sn.heatmap(df_cm, annot=True, cmap='Blues',linecolor='black',square=True)
    g.set_xticklabels(labels, rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def GetAcc(net, epoch, dataLoader, loader_string, terrain_flag=True):
    # go into eval mode
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataLoader, 0):
            inputs = data['tactile_data']
            features = data['custom_features']
            labels_class = data['class']
            labels_terrain = data['terrain']

            inputs = inputs.float()
            features = features.float()
            labels_class = labels_class.long()
            labels_terrain = labels_terrain.long()

            inputs = inputs.to(device)
            features = features.to(device)
            labels_class = labels_class.to(device)
            labels_terrain = labels_terrain.to(device)

            outputs = net(inputs, features)
            _, prediction = torch.max(outputs.data, 1)
            if terrain_flag:
                total += labels_terrain.size(0)
                correct += (prediction == labels_terrain).sum().item()
            else:
                total += labels_class.size(0)
                correct += (prediction == labels_class).sum().item()
            
            
    acc = 100*correct/total
    print( loader_string + '[%d, %5d] accuracy: %.3f' %
          (epoch + 1, i + 1, acc))
    
    return acc

def GetDicts(model, dataLoader, terrain_flag=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct_dict_terrain = {}
    incorrect_dict_terrain = {}
    correct_dict_class = {}
    incorrect_dict_class = {}

    for i in range(8):
        incorrect_dict_terrain[i] = {}

    for i in range(4):
        incorrect_dict_class[i] = {}
        
    correct_terrain = 0
    correct_class = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataLoader, 0):
            model.eval()

            inputs = data['tactile_data']
            features = data['custom_features']
            labels_class = data['class']
            labels_terrain = data['terrain']

            inputs = inputs.float()
            features = features.float()
            labels_class = labels_class.long()
            labels_terrain = labels_terrain.long()

            inputs = inputs.to(device)
            features = features.to(device)
            labels_class = labels_class.to(device)
            labels_terrain = labels_terrain.to(device)

            outputs = model(inputs, features)
            if terrain_flag:
                _, prediction_terrain = torch.max(outputs.data, 1)
                total += labels_terrain.size(0)
                correct_terrain += (prediction_terrain == labels_terrain).sum().item()
                
                for pred_terrain,label_terrain,label_class in zip(prediction_terrain,labels_terrain,labels_class):
                    pred_terrain = int(pred_terrain)
                    label_terrain = int(label_terrain)

                    pred_class = dp.Terrain2Class(pred_terrain)
                    label_class = int(label_class)

                    if pred_terrain == label_terrain:
                        if label_terrain not in correct_dict_terrain:
                            correct_dict_terrain[label_terrain] = 1
                        else:
                            correct_dict_terrain[label_terrain] +=1
                    else:
                        incorrect_dict_label = incorrect_dict_terrain[label_terrain]
                        if pred_terrain not in incorrect_dict_label:
                            incorrect_dict_label[pred_terrain] = 1
                        else:
                            incorrect_dict_label[pred_terrain] +=1
                    if pred_class == label_class:
                        if label_class not in correct_dict_class:
                            correct_dict_class[label_class] = 1
                        else:
                            correct_dict_class[label_class] +=1
                    else:
                        incorrect_dict_label = incorrect_dict_class[label_class]
                        if pred_class not in incorrect_dict_label:
                            incorrect_dict_label[pred_class] = 1
                        else:
                            incorrect_dict_label[pred_class] +=1
            else:
                _, prediction_class = torch.max(outputs.data, 1)
                total += labels_class.size(0)
                correct_class += (prediction_class == labels_class).sum().item()

                for pred_class,label_class in zip(prediction_class,labels_class):
                    
                    pred_class = int(pred_class)
                    label_class = int(label_class)

                    if pred_class == label_class:
                        if label_class not in correct_dict_class:
                            correct_dict_class[label_class] = 1
                        else:
                            correct_dict_class[label_class] +=1
                    else:
                        incorrect_dict_label = incorrect_dict_class[label_class]
                        if pred_class not in incorrect_dict_label:
                            incorrect_dict_label[pred_class] = 1
                        else:
                            incorrect_dict_label[pred_class] +=1
          


    if terrain_flag:                      
        print('Accuracy of the network on the validation set for TERRAIN: %d %%' % (
            100 * correct_terrain / total))
        return correct_dict_terrain, incorrect_dict_terrain, correct_dict_class, incorrect_dict_class
    else:
        print('Accuracy of the network on the validation set for CLASS: %d %%' % (
            100 * correct_class / total))
        return correct_dict_class, incorrect_dict_class
