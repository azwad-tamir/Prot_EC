import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Loading dataset from csv file:
df_train = pd.read_csv('./train_df')

# Preprocessing labels: output-labels_OHE
EC_total1 = list(df_train['EC_num'].values)
seq = list(df_train['sequence'].values)
text = []
labels = []

for num in range(0,len(seq)):
    if EC_total1[num] != '[]':
        # text.append(seq[num])
        comment = seq[num].replace(' ', '')
        text.append(' '.join(comment))
        labels.append(int(EC_total1[num][2])-1)

labels_OHE = []
label_num = len(list(set(labels)))
for i in range(0, len(labels)):
    l_arr = np.zeros((label_num), int)
    l_arr[labels[i]] = int(1)
    labels_OHE.append(l_arr)


# for i in range(0, len(labels)):
# for comment in seq:
#     comment = comment.replace(' ','')
#     text.append(' '.join(comment))


# labels = []
# for EC_num in EC_total1:
#     if EC_num == '[]':
#         labels.append(0)
#     else:
#         labels.append(1)

text_train, text_test, labels_train, labels_test = train_test_split(text, labels_OHE, stratify=labels, test_size=0.1)

# labels_train_1he = []
# labels_test_1he = []
# for label in labels_train:
#     l_arr = np.zeros((label_num), int)
#     l_arr[label] = int(1)
#     str_arr = ""
#     for i in l_arr:
#         str_arr += (str(i))
#
#     # text.append(','.join(comment))
#     str_arr1 = ','.join(str_arr)
#     labels_train_1he.append(str_arr1)


# for label in labels_test:
#     l_arr = np.zeros((label_num), int)
#     l_arr[label] = int(1)
#     str_arr = ""
#     for i in l_arr:
#         str_arr += (str(i))
#
#     # text.append(','.join(comment))
#     str_arr1 = ','.join(str_arr)
#     labels_test_1he.append(str_arr1)

dict_train = {'text':text_train,'labels':labels_train}
dict_test = {'text':text_test,'labels':labels_test}
dict_all = {'train':dict_train, 'test':dict_test}
df_train = pd.DataFrame(dict_train)
df_test = pd.DataFrame(dict_test)
df_train.to_csv('df_train_fn_1he.csv', index = False)
df_test.to_csv('df_test_fn_1he.csv', index = False)

# sum=0
# for x in labels_test:
#     sum = sum + x
