import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, BertModel, BertTokenizer, BertForMaskedLM, BertForPreTraining, \
    BertForSequenceClassification, AutoModelForSequenceClassification
import torch
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import TrainingArguments
import pandas as pd
from datasets import load_metric
import random
from torch import nn
from transformers import Trainer
import os
from transformers import AutoConfig, EarlyStoppingCallback
from transformers.models.bert import BertPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import IntervalStrategy
import shutil
import time
from sklearn import preprocessing
import pickle
from collections import Counter
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
import seaborn as sn
import matplotlib.pyplot as plt

with open('./Accuracy_matrics_random_EC1_EX_tune.pkl', 'rb') as f:
    EC1_accuracy_list = pickle.load(f)

with open('./Accuracy_matrics_random_EC2_EX_tune.pkl', 'rb') as f:
    EC2_accuracy_list = pickle.load(f)

with open('./Accuracy_matrics_random_EC3_EX_tune1.pkl', 'rb') as f:
    EC3_accuracy_list = pickle.load(f)

with open('./Accuracy_matrics_random_EC4_EX_tune1.pkl', 'rb') as f:
    EC4_accuracy_list = pickle.load(f)

# all_predictions_train_EC1 = EC1_accuracy_list[0]
# all_labels_train_EC1 = EC1_accuracy_list[1]
# all_logits_train_EC1 = EC1_accuracy_list[2]
# accuracy_train_EC1 = EC1_accuracy_list[3]
all_predictions_eval_EC1 = EC1_accuracy_list[4]
all_labels_eval_EC1 = EC1_accuracy_list[5]
all_logits_eval_EC1 = EC1_accuracy_list[6]
accuracy_eval_EC1 = EC1_accuracy_list[7]
recall_EC1 = recall_score(all_labels_eval_EC1, all_predictions_eval_EC1, average="weighted")
precision_EC1 = precision_score(all_labels_eval_EC1, all_predictions_eval_EC1, average="weighted")
F1_score_EC1 = f1_score(all_labels_eval_EC1, all_predictions_eval_EC1, average="weighted")
print("EC1 accuracy: ", accuracy_eval_EC1)
print("EC1 Precision: ", precision_EC1)
print("EC1 Recall: ", recall_EC1)
print("EC1 F1_score: ", F1_score_EC1)

all_predictions_eval_EC2 = EC2_accuracy_list[4]
all_labels_eval_EC2 = EC2_accuracy_list[5]
all_logits_eval_EC2 = EC2_accuracy_list[6]
accuracy_eval_EC2 = EC2_accuracy_list[7]
recall_EC2 = recall_score(all_labels_eval_EC2, all_predictions_eval_EC2, average="weighted")
precision_EC2 = precision_score(all_labels_eval_EC2, all_predictions_eval_EC2, average="weighted")
F1_score_EC2 = f1_score(all_labels_eval_EC2, all_predictions_eval_EC2, average="weighted")
print("EC2 accuracy: ", accuracy_eval_EC2)
print("EC2 Precision: ", precision_EC2)
print("EC2 Recall: ", recall_EC2)
print("EC2 F1_score: ", F1_score_EC2)

all_predictions_eval_EC3 = EC3_accuracy_list[4]
all_labels_eval_EC3 = EC3_accuracy_list[5]
all_logits_eval_EC3 = EC3_accuracy_list[6]
accuracy_eval_EC3 = EC3_accuracy_list[7]
recall_EC3 = recall_score(all_labels_eval_EC3, all_predictions_eval_EC3, average="weighted")
precision_EC3 = precision_score(all_labels_eval_EC3, all_predictions_eval_EC3, average="weighted")
F1_score_EC3 = f1_score(all_labels_eval_EC3, all_predictions_eval_EC3, average='weighted')
print("EC3 accuracy: ", accuracy_eval_EC3)
print("EC3 Precision: ", precision_EC3)
print("EC3 Recall: ", recall_EC3)
print("EC3 F1_score: ", F1_score_EC3)

all_predictions_eval_EC4 = EC4_accuracy_list[4]
all_labels_eval_EC4 = EC4_accuracy_list[5]
all_logits_eval_EC4 = EC4_accuracy_list[6]
accuracy_eval_EC4 = EC4_accuracy_list[7]
recall_EC4 = recall_score(all_labels_eval_EC4, all_predictions_eval_EC4, average="weighted")
precision_EC4 = precision_score(all_labels_eval_EC4, all_predictions_eval_EC4, average="weighted")
F1_score_EC4 = f1_score(all_labels_eval_EC4, all_predictions_eval_EC4, average='weighted')
print("EC4 accuracy: ", accuracy_eval_EC4)
print("EC4 Precision: ", precision_EC4)
print("EC4 Recall: ", recall_EC4)
print("EC4 F1_score: ", F1_score_EC4)


# Plotting Confusion matrix:

# Precsion and recall:
print("\n\nENSEMBLE MODEL:  ")
print(classification_report(all_labels_eval_EC1, all_predictions_eval_EC1,digits=4))
# Confusion Matrix:
y_true = all_labels_eval_EC1
y_pred = all_predictions_eval_EC1
data = confusion_matrix(y_true, y_pred)
labels_list = []
for i in range(0,len(np.unique(y_true))):
    labels_list.append(str(le_1.inverse_transform([np.unique(y_true)[i]])[0]))

# a = le_2.inverse_transform([0])
# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm = pd.DataFrame(data, columns=labels_list, index = labels_list)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
# plt.title("Confusion matrix for EC1 on the Random split")
plt.show()

# Plotting Confusion matrix:

# Precsion and recall:
print("\n\nENSEMBLE MODEL:  ")
print(classification_report(all_labels_eval_EC2, all_predictions_eval_EC2,digits=4))
# Confusion Matrix:
y_true = all_labels_eval_EC2
y_pred = all_predictions_eval_EC2
data = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
labels_list = []
for i in range(0,len(np.unique(y_true))):
    labels_list.append(str(le_2.inverse_transform([np.unique(y_true)[i]])[0]))

# a = le_2.inverse_transform([0])
# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm = pd.DataFrame(data, columns=labels_list, index = labels_list)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (15,10))
sn.set(font_scale=1.0)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='d')# font size
# plt.title("Confusion matrix for EC2 for the Random split")
plt.show()

# Plotting Confusion matrix:

# Precsion and recall:
print("\n\nENSEMBLE MODEL:  ")
print(classification_report(all_labels_eval_EC3, all_predictions_eval_EC3,digits=4))
# Confusion Matrix:
y_true = all_labels_eval_EC3
y_pred = all_predictions_eval_EC3
data = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
labels_list = []
for i in range(0,len(np.unique(y_true))):
    labels_list.append(str(le_3.inverse_transform([np.unique(y_true)[i]])[0]))

# a = le_2.inverse_transform([0])
# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm = pd.DataFrame(data, columns=labels_list, index = labels_list)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.0)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 9}, fmt='d')# font size
# plt.title("Confusion matrix for EC3 for the Random split")
plt.show()

# Sequence length analysis:
ECX_labl = all_labels_eval_EC4
ECX_pred = all_predictions_eval_EC4
ECX_seq = text_dev
pred_status = []
ECX_seq_len = []
seq_len_accuracies = np.zeros((18000))
seq_len_freq = np.zeros((18000))
for i in range(0,len(ECX_seq)):
    seq_len_freq[len(ECX_seq[i])] += 1

for i in range(0,len(ECX_seq)):
    ECX_seq_len.append(len(ECX_seq[i]))
    if ECX_labl[i] == ECX_pred[i]:
        pred_status.append(1)
        seq_len_accuracies[len(ECX_seq[i])] += 1
    else:
        pred_status.append(0)

#
# bins = list(range(0,2100,100))
# plt.xlabel("Sequence Length")
# plt.ylabel("Number of Sequences")
# plt.hist(accuracy_bin, bins=bins)

x_value = []
y_value = []
accuracy_bin = np.zeros((2000))
for i in range(0,2000,100):
    print(i,"   ",i+100)
    accuracy_bin[i+50]=np.sum(seq_len_accuracies[i:(i+100)])/np.sum(seq_len_freq[i:(i+100)])*100
    x_value.append(i)
    y_value.append(np.sum(seq_len_accuracies[i:(i+100)])/np.sum(seq_len_freq[i:(i+100)])*100)
    x_value.append(i+99.99)
    y_value.append(np.sum(seq_len_accuracies[i:(i + 100)]) / np.sum(seq_len_freq[i:(i + 100)]) * 100)


x_value4 = x_value
y_value4 = y_value

plt.plot(x_value1,y_value1, label="EC1")
plt.plot(x_value2,y_value2, label="EC2")
plt.plot(x_value3,y_value3, label="EC3")
plt.plot(x_value4,y_value4, label="EC4")
plt.ylim(70,101)
plt.xlabel('Sequence Length')
plt.ylabel('% Accuracy')
# plt.title('Averaged ROC curve for each EC_num for random split')
plt.legend(loc="lower right")



plt.show()


# x_value = []
# y_value = []
# for i in range(0,len(seq_len_freq)):
#     if seq_len_freq[i]!=0:
#         x_value.append(i)
#         y_value.append(seq_len_accuracies[i]/seq_len_freq[i]*100)
#
# for i in range(0,len(x_value)):
#     accuracy_bin[x_value[i]] = y_value[i]

