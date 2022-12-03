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
import glob
from collections import Counter
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

mode = 0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
EC1_STS_models_list = []
for i in range(0,21):
    EC1_STS_models_list.append("models_STS_EC1_"+str(i))

accuracy_matrics_Random_EC1_STS_list = []
Fractions = [1.0, 0.8499973661905058, 0.7224970429502497, 0.6141181766376309, 0.5219970980208118, 0.4436963361315564,
             0.37713757584174157, 0.3205633479070792, 0.2724747752881627, 0.23160284068325807, 0.19686049908295541,
             0.16732831153656447, 0.1422257126849054, 0.11378057014792432, 0.08533542761094325, 0.059734799327660265,
             0.03882714068852569, 0.02329245341748754, 0.012809891630711176, 0.006402551443088165, 0.002878035465442025]
for EC1_STS_models in range(0,21):

    ########################################################################################################################
    ########################################################################################################################
    #                                                    EC1_Random_test                                                   #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################### Data Preparation: ##########################################
    # Loading dataset from csv file:
    df_train = pd.read_csv('./random_train_df')
    df_dev = pd.read_csv('./random_dev_df')
    df_test = pd.read_csv('./random_test_df')

    # Preprocessing training labels: output-labels_OHE
    EC_total1 = list(df_train['EC_num'].values)
    seq = list(df_train['sequence'].values)
    seq_clean = []
    EC_labels = []
    EC1_labels = []
    EC2_labels = []
    EC3_labels = []
    EC4_labels = []
    for i in range(0,len(EC_total1)):
        EC_labels_temp = []
        if EC_total1[i] != '[]':
            EC_total_struc = EC_total1[i].replace('[', '').replace(']', '').replace(' ', '').replace('\'', '').split(',')
            # Mining first EC value:
            EC1 = EC_total_struc[0].split('.')[0]
            EC1_labels.append(EC1)
            EC_labels_temp.append(EC1)

            # Mining second EC value:
            EC2 = 'x'
            for a in EC_total_struc:
                # EC2 = a.split('.')[1]
                if a.split('.')[1] != '-':
                    EC2 = a.split('.')[1]
                    break

            EC2_labels.append(EC2)
            EC_labels_temp.append(EC2)

            # Mining third EC value:
            EC3 = 'x'
            for a in EC_total_struc:
                # EC2 = a.split('.')[1]
                if a.split('.')[2] != '-':
                    EC3 = a.split('.')[2]
                    break

            EC3_labels.append(EC3)
            EC_labels_temp.append(EC3)

            # Mining fourth EC value:
            EC4 = 'x'
            for a in EC_total_struc:
                # EC2 = a.split('.')[1]
                if a.split('.')[3] != '-':
                    EC4 = a.split('.')[3]
                    break

            EC4_labels.append(EC4)
            EC_labels_temp.append(EC4)

            EC_labels.append(EC_labels_temp)
        else:
            EC1_labels.append('x')
            EC2_labels.append('x')
            EC3_labels.append('x')
            EC4_labels.append('x')
            EC_labels.append([])

    le_1 = preprocessing.LabelEncoder()
    le_1.fit(EC1_labels)
    EC1_labels_encoded = le_1.transform(EC1_labels)
    # le_2 = preprocessing.LabelEncoder()
    # le_2.fit(EC2_labels)
    # EC2_labels_encoded = le_2.transform(EC2_labels)
    # le_3 = preprocessing.LabelEncoder()
    # le_3.fit(EC3_labels)
    # EC3_labels_encoded = le_3.transform(EC3_labels)
    # le_2.inverse_transform()


    """
    Total samples: 438522
    Total blank EC_num: 229699
    Blank EC1: 229699
    Blank EC2: 231657
    Blank EC3: 237544
    Blank EC4: 260220
    """

    # count=0
    # for a in EC4_labels:
    #     if a == 'x':
    #         count+=1

    text = []
    labels = []
    check = []
    for i in range(0,len(EC1_labels_encoded)):
        if EC1_labels[i] != 'x':
            comment = ' '.join(seq[i])
            # comment = str(EC2_labels_encoded[i]) + " " + comment
            text.append(comment)
            labels.append(EC1_labels_encoded[i])


    n=100
    text_3 = random.sample(text, n)
    labels_3 = random.sample(labels, n)
    ####################################################

    labels_OHE = []
    label_num = len(list(set(labels_3)))
    for i in range(0, len(labels_3)):
        l_arr = np.zeros((label_num), int)
        l_arr[labels_3[i]] = int(1)
        labels_OHE.append(l_arr)


    counts = np.zeros((label_num), int)
    for i in range(0, len(labels_3)):
        counts[labels_3[i]] += 1

    text_train = text_3
    labels_train = labels_OHE



    # Preprocessing dev labels: output-labels_OHE
    EC_total1 = list(df_test['EC_num'].values)
    seq = list(df_test['sequence'].values)
    seq_clean = []
    EC_labels = []
    EC1_labels = []
    EC2_labels = []
    EC3_labels = []
    EC4_labels = []
    for i in range(0,len(EC_total1)):
        EC_labels_temp = []
        if EC_total1[i] != '[]':
            EC_total_struc = EC_total1[i].replace('[', '').replace(']', '').replace(' ', '').replace('\'', '').split(',')
            # Mining first EC value:
            EC1 = EC_total_struc[0].split('.')[0]
            EC1_labels.append(EC1)
            EC_labels_temp.append(EC1)

            # Mining second EC value:
            EC2 = 'x'
            for a in EC_total_struc:
                # EC2 = a.split('.')[1]
                if a.split('.')[1] != '-':
                    EC2 = a.split('.')[1]
                    break

            EC2_labels.append(EC2)
            EC_labels_temp.append(EC2)

            # Mining third EC value:
            EC3 = 'x'
            for a in EC_total_struc:
                # EC2 = a.split('.')[1]
                if a.split('.')[2] != '-':
                    EC3 = a.split('.')[2]
                    break

            EC3_labels.append(EC3)
            EC_labels_temp.append(EC3)

            # Mining fourth EC value:
            EC4 = 'x'
            for a in EC_total_struc:
                # EC2 = a.split('.')[1]
                if a.split('.')[3] != '-':
                    EC4 = a.split('.')[3]
                    break

            EC4_labels.append(EC4)
            EC_labels_temp.append(EC4)

            EC_labels.append(EC_labels_temp)
        else:
            EC1_labels.append('x')
            EC2_labels.append('x')
            EC3_labels.append('x')
            EC4_labels.append('x')
            EC_labels.append([])

    # Calculating x structure for successive EC numbers:
    index_hasEC1_notEC2 = []
    index_hasEC1_notEC3 = []
    index_hasEC2_notEC3 = []
    index_hasEC1_notEC4 = []
    index_hasEC2_notEC4 = []
    index_hasEC3_notEC4 = []

    index1 = index2 = index3= index4 = 0
    for i in range(0,len(EC1_labels)):
        if EC1_labels[i] != 'x':
            if EC2_labels[i] == 'x':
                index_hasEC1_notEC2.append(index1)
            if EC3_labels[i] == 'x':
                index_hasEC1_notEC3.append(index1)
            if EC4_labels[i] == 'x':
                index_hasEC1_notEC4.append(index1)
            index1+=1
        if EC2_labels[i] != 'x':
            if EC3_labels[i] == 'x':
                index_hasEC2_notEC3.append(index2)
            if EC4_labels[i] == 'x':
                index_hasEC2_notEC4.append(index2)
            index2+=1
        if EC3_labels[i] != 'x':
            if EC4_labels[i] == 'x':
                index_hasEC3_notEC4.append(index3)
            index3+=1


    # le_1 = preprocessing.LabelEncoder()
    # le_1.fit(EC1_labels)
    EC1_labels_encoded = le_1.transform(EC1_labels)
    # le_2 = preprocessing.LabelEncoder()
    # le_2.fit(EC2_labels)

    # EC2_labels_encoded = le_2.transform(EC2_labels) ###Fix needed for EC2 classification

    # le_3 = preprocessing.LabelEncoder()
    # le_3.fit(EC3_labels)
    # EC3_labels_encoded = le_3.transform(EC3_labels)
    # le_2.inverse_transform()


    """
    Total samples: 438522
    Total blank EC_num: 229699
    Blank EC1: 229699
    Blank EC2: 231657
    Blank EC3: 237544
    Blank EC4: 260220
    """

    count=0
    for a in EC1_labels:
        if a == 'x':
            count+=1

    text = []
    labels = []
    check = []
    for i in range(0,len(EC1_labels_encoded)):
        if EC1_labels[i] != 'x':
            comment = ' '.join(seq[i])
            # comment = str(EC2_labels_encoded[i]) + " " + comment
            text.append(comment)
            labels.append(EC1_labels_encoded[i])


    text_3 = text
    labels_3 = labels

    # sum=0
    # EC1_temp = []
    # for i in range(0,len(EC1_labels)):
    #     if EC1_labels[i] != 'x':
    #         EC1_temp.append(EC1_labels[i])
    #
    # for i in range(0,len(EC1_temp)):
    #     if EC1_temp[i] == pred_orig[i]:
    #         sum+=1
    ####################################################

    labels_OHE = []
    label_num = len(list(set(labels_3)))

    for i in range(0, len(labels_3)):
        l_arr = np.zeros((label_num), int)
        l_arr[labels_3[i]] = int(1)
        labels_OHE.append(l_arr)


    counts = np.zeros((label_num), int)
    for i in range(0, len(labels_3)):
        counts[labels_3[i]] += 1

    text_dev = text_3
    labels_dev = labels_OHE

    # text_train, text_test, labels_train, labels_test = train_test_split(text_3, labels_OHE, stratify=labels_3, test_size=0.1)
    # # text_test1, text_test2, labels_test1, labels_test2 = train_test_split(text_test, labels_test, test_size=0.1)

    dict_train = {'text':text_train,'labels':labels_train}
    dict_test = {'text':text_dev,'labels':labels_dev}
    ########################################################################################################################


    ########################################################################################################################
    ############################################### Building Model: ########################################################

    class MyBertForSequenceClassification(BertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.config = config

            self.bert = BertModel(config)
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)
            if config.intermediate_hidden_size != 0:
                self.intermediate_classifier = nn.Linear(config.hidden_size, config.intermediate_hidden_size)
                self.classifier = nn.Linear(config.intermediate_hidden_size, config.num_labels)
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            self.init_weights()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.config.use_pooler == 1:
                pooled_output = outputs[1]
            elif self.config.use_mean == 1:
                # print(attention_mask.shape)
                # token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
                pooled_output = torch.sum(outputs[0] * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
                # print(pooled_output.shape)
                # pooled_output = torch.mean(outputs[0], dim=1)
            else:
                pooled_output = outputs[0][:, 0]

            pooled_output = self.dropout(pooled_output)
            if config.intermediate_hidden_size != 0:
                intermediate_output = self.intermediate_classifier(pooled_output)
                logits = self.classifier(intermediate_output)
            else:
                logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    # loss_fct = BCEWithLogitsLoss()
                    loss_fct = CrossEntropyLoss()
                    # print(list(logits.size()))
                    # print(list(labels.size()))
                    # print(logits.dtype)
                    # print(labels.dtype)
                    labels = labels.type(torch.float32)
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


    def tokenize_function(example):

        return tokenizer(example["text"], add_special_tokens=True, truncation=True, max_length=1024)



    def compute_metrics(eval_preds):
        # print("PROBLEM!!!")
        metric = load_metric("accuracy")
        # sdfdf
        # metric = load_metric("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        labels_real = np.argmax(labels, axis=1)
        return metric.compute(predictions=predictions, references=labels_real)

    model_type = "Rostlab/prot_bert_bfd"
    # model_type = "Rostlab/prot_bert"
    # model_type = "bert-base-cased"
    # model_type = "bert-base-uncased"
    # model_type = "distilbert-base-uncased"

    # do_lower_case = True
    do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=do_lower_case)
    # elif (a==str(0) or a==str(1) or a==str(2) or a==str(3) or a==str(4) or a==str(5) or a==str(6)):
    # tokenizer.add_tokens(["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"])



    ##########################################################################################################################
    # import torch
    # from transformers import BertTokenizer, BertModel
    #
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # model = BertModel.from_pretrained("bert-base-cased")
    #
    # print(len(tokenizer))  # 28996
    # tokenizer.add_tokens(["NEW_TOKEN1"])
    # print(len(tokenizer))  # 28997
    #
    # model.resize_token_embeddings(len(tokenizer))
    # # The new vector is added at the end of the embedding matrix
    #
    # print(len(model.embeddings.word_embeddings.weight[-1, :]))
    # # Randomly generated matrix
    #
    # model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])
    #
    # print(len(model.embeddings.word_embeddings.weight[-1, :]))
    # # outputs a vector of zeros of shape [768]

    ########################################################################################################################

    # model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd")
    config = AutoConfig.from_pretrained(model_type)
    # subject = 'secreted'
    # subject = 'iamppred'
    # subject = 'ampscanner'
    # subject = 'iamp2l'
    # subject = 'ampep'
    # subject = 'secproct'
    # subject = 'hemolythic'
    # subject = 'hlppredfuse'
    # subject = 'rnnamp'
    subject = 'EC_num'

    config.classifier_dropout = 0
    config.hidden_dropout_prob = 0
    # config.hidden_size = 1024

    mode_hidden_map = {0:0, 1:32, 2:128, 3:1024}
    # config.intermediate_hidden_size = 1024
    # config.intermediate_hidden_size = 128
    # config.intermediate_hidden_size = 32
    # config.intermediate_hidden_size = 0
    config.intermediate_hidden_size = mode_hidden_map[mode]


    ########################################################################################################################
    ##################################################### Training: ##########################################################
    # num_epochs = 1
    # num_epochs = 10
    num_epochs = 10
    # num_epochs = 100

    config.use_pooler = 0
    config.use_mean = 1


    # Enabling multilabel classification:
    config.num_labels = label_num
    config.problem_type = "multi_label_classification"

    # freeze_positional = 0
    # freeze_non_positional = 0
    # freeze_attention = 0
    # freeze_layer_norm = 0
    # freeze_pooler = 0

    # freeze_positional = 1
    # freeze_non_positional = 1
    # freeze_attention = 1
    # freeze_layer_norm = 1
    # freeze_pooler = 1

    freeze_positional = 0
    freeze_non_positional = 1
    freeze_attention = 1
    freeze_layer_norm = 0
    freeze_pooler = 0

    transfer = 0
    random_init = 0
    if random_init:
        transfer = 0
    if subject == 'secreted':
        transfer = 0

    early_stopping = 1
    patience = 10
    if early_stopping:
        create_validation_split = 1
    else:
        create_validation_split = 0
    ten_fold = 0

    # monitor_value = ''
    # initial_lr = 5e-6
    # initial_lr = 5e-5
    initial_lr = 5e-4
    if subject == 'iamp2l' or subject == 'iamppred':
        batch_size = 32
    if subject == 'secreted' or subject == 'ampscanner' or subject == 'hemolythic' or subject == 'hemolythic_2021' or subject == 'hlppredfuse' or subject == 'rnnamp':
        batch_size = 16
    if subject == 'ampep' or subject == 'secproct':
        batch_size = 8
    if subject == 'secproct':
        batch_size = 4
    if subject == 'EC_num':
        batch_size = 1
    balanced_loss = 0

    # batch_size = 4
    losses_all = []
    fold_range = [0]
    if ten_fold:
        fold_range = range(10)

    for fold in fold_range:
        model = None
        dataset = None
        if transfer == 0:
            if random_init:
                model = MyBertForSequenceClassification(config=config)
            else:
                model = MyBertForSequenceClassification.from_pretrained(model_type, config=config)
        if transfer == 1:
            results_df = pd.read_csv('results/training_results.csv')
            results_df = results_df[results_df['subject'] == 'secreted']
            results_df = results_df[results_df['hidden_layer_size'] == config.intermediate_hidden_size]
            results_df = results_df[results_df['usemean'] == config.use_mean]
            results_df = results_df[results_df['usepooler'] == config.use_pooler]
            secreted_model_dir = results_df.iloc[0]['save_dir']
            model = MyBertForSequenceClassification.from_pretrained(secreted_model_dir, config=config)

        # Loading Checkpoint:
        # model = MyBertForSequenceClassification.from_pretrained("./models/0706-172025/checkpoint-4350")

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # model.to(device)
        # dataset = load_dataset('text', data_files={'train': 'data/uniparc_peptides_spaced.txt'}, cache_dir='data/processed_datasets')
        # dataset = load_dataset('csv', data_files={'train': 'old/input/amp.csv'},
        #                        # cache_dir='data/processed_datasets',
        #                        delimiter=',',
        #                        # task="text-classification"
        #                        )

        # CSV file loader, 2 columns, 'text' and 'labels'. Text is spaced capital sequences.
        if subject == 'iamp2l':
            dataset = load_dataset('csv', data_files={'train': 'data/iamp2l/iamp2l_raw.csv', 'test': 'data/iamp2l/independent/independent.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )
        elif subject == 'iamppred':
            dataset = load_dataset('csv', data_files={'train': 'data/iamppred/old/train.csv', 'test': 'data/iamppred/old/test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter='\t',
                                   # task="text-classification"
                                   )
        elif subject == 'secreted':
            dataset = load_dataset('csv', data_files={'train': 'data/swissprot/secreted_all_cleaned_all_train.csv', 'test': 'data/swissprot/secreted_all_cleaned_all_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )

        elif subject == 'ampscanner':
            dataset = load_dataset('csv', data_files={'train': 'data/ampscanner/ampscanner_train.csv',
                                                      'test': 'data/ampscanner/ampscanner_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )

        elif subject == 'ampep':
            dataset = load_dataset('csv', data_files={'train': 'data/ampep/ampep.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   )

        elif subject == 'secproct':
            dataset = load_dataset('csv', data_files={'train': 'data/secproct/blood_train.csv',
                                                      'test': 'data/secproct/blood_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )
        elif subject == 'hemolythic':
            dataset = load_dataset('csv', data_files={'train': 'data/hemolythic/hemolythic_train.csv',
                                                      'test': 'data/hemolythic/hemolythic_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )
        elif subject == 'hemolythic_2021':
            dataset = load_dataset('csv', data_files={'train': 'data/hemolythic_2021/hemolythic_2021_train.csv',
                                                      'test': 'data/hemolythic_2021/hemolythic_2021_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )
        elif subject == 'hlppredfuse':
            dataset = load_dataset('csv', data_files={'train': 'data/hlppredfuse/hlppredfuse_train.csv',
                                                      'test': 'data/hlppredfuse/hlppredfuse_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )
        elif subject == 'rnnamp':
            dataset = load_dataset('csv', data_files={'train': 'data/rnnamp/rnnamp_train.csv',
                                                      'test': 'data/rnnamp/rnnamp_test.csv'},
                                   cache_dir='data/processed_datasets',
                                   delimiter=',',
                                   # task="text-classification"
                                   )

        elif subject == 'EC_num':
            # dataset = Dataset.from_pandas(df)
            dataset_train = Dataset.from_dict(dict_train, split='train')
            dataset_test = Dataset.from_dict(dict_test, split='test')

            # dataset_test2 = Dataset.from_dict(dict_test2, split='test')
            # dataset = load_dataset('dict')
            # dataset  = load_dataset('csv', data_files={'train': 'df_test_temp.csv',
            #                                           'test': 'df_test_temp.csv'},
            #                        cache_dir='data/processed_datasets',
            #                        delimiter=',',
            #                        # task="text-classification"
            #                        )

        # reload = False
        # if not reload:
        #     # Find the indices for train, validation, and test splits
        #     random.seed(42)
        #     data_len = len(dataset['train'])
        #     randomlist = list(range(data_len))
        #     random.shuffle(randomlist)
        #     valid_begin = int(fold * 0.1 * data_len)
        #     valid_end = int((fold + 1) * 0.1 * data_len)
        #     test_begin = valid_end
        #     test_end = int((fold + 2) * 0.1 * data_len)
        #     if fold == 9:
        #         test_begin = 0
        #         test_end = int(0.1 * data_len)
        #     test_indices = randomlist[test_begin:test_end]
        #     valid_indices = randomlist[valid_begin:valid_end]
        #
        #     # Create the validation split
        #     dataset['validation'] = dataset['train'].select(valid_indices)
        #     if ten_fold:
        #         # In ten fold cross validation, isolate the test split
        #         dataset['test'] = dataset['train'].select(test_indices)
        #         if create_validation_split:
        #             train_indices = [ind for ind in randomlist if ind not in test_indices + valid_indices]
        #             dataset['train'] = dataset['train'].select(train_indices)
        #         else:
        #             # If no validation split is needed (for no early stopping), do not isolate it from the train split
        #             train_indices = [ind for ind in randomlist if ind not in test_indices]
        #             dataset['train'] = dataset['train'].select(train_indices)
        #     else:
        #         if create_validation_split:
        #             train_indices = [ind for ind in randomlist if ind not in valid_indices]
        #             dataset['train'] = dataset['train'].select(train_indices)
        #     del randomlist


        tokenized_datasets_train = dataset_train.map(tokenize_function, batched=True)
        tokenized_datasets_test = dataset_test.map(tokenize_function, batched=True)
        # all_data = tokenize_function(dataset_test)
        # tokenized_datasets_test = copy.deepcopy(dataset_test)
        # tokenized_datasets_test = tokenized_datasets_test.add_column('input_ids', all_data[0])
        # tokenized_datasets_test = tokenized_datasets_test.add_column('token_type_ids', all_data[2])
        # tokenized_datasets_test = tokenized_datasets_test.add_column('attention_mask', all_data[1])

        # all_data = tokenize_function(dataset_train)
        # tokenized_datasets_train = copy.deepcopy(dataset_train)
        # tokenized_datasets_train = tokenized_datasets_train.add_column('input_ids', all_data[0])
        # tokenized_datasets_train = tokenized_datasets_train.add_column('token_type_ids', all_data[2])
        # tokenized_datasets_train = tokenized_datasets_train.add_column('attention_mask', all_data[1])

        # tokenized_datasets_test2 = dataset_test2.map(tokenize_function, batched=True)
        # tokenized_datasets_protein = dataset_protein.map(tokenize_function, batched=True)
        tokenized_datasets_train.set_format("torch", columns=['input_ids'])
        tokenized_datasets_test.set_format("torch", columns=['input_ids'])
        # tokenized_datasets_test2.set_format("torch", columns=['input_ids'])
        # tokenized_datasets.save_to_disk('data/processed_datasets/peptides')
        # tokenized_datasets.save_to_disk('data/processed_datasets/'+subject+str(fold))
        # tokenized_datasets.save_to_disk('data/processed_datasets/'+ subject)

        # else:
        #     # tokenized_datasets = load_from_disk('data/processed_datasets/peptides')
        #     tokenized_datasets = load_from_disk('data/processed_datasets/'+subject+str(fold))
        #     # tokenized_datasets = load_from_disk('data/processed_datasets/secreted')

        tokenized_datasets_train = tokenized_datasets_train.remove_columns('text')
        tokenized_datasets_test = tokenized_datasets_test.remove_columns('text')
        # tokenized_datasets_test2 = tokenized_datasets_test2.remove_columns('text')
        # for c in tokenized_datasets.column_names['train']:
        #     if c in ['keyword', 'length']:
        #         tokenized_datasets = tokenized_datasets.remove_columns(c)

        tokenized_datasets_train.set_format("torch")
        tokenized_datasets_test.set_format("torch")
        # tokenized_datasets_test2.set_format("torch")
        # tokenized_datasets_protein.set_format("torch", columns=['input_ids'])
        # print(tokenized_datasets['train'][1])
        # tokenized_datasets = s.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

        timestr = time.strftime("%m%d-%H%M%S")
        save_dir = 'models_temp/' + timestr + '/'
        while os.path.isdir(save_dir):
            timestr = timestr.split('-')[0] + '-' + timestr.split('-')[1][:4] + str(int(timestr.split('-')[1][4:] + random.randint(1,60)))
            save_dir = 'models_temp/' + timestr + '/'

        os.makedirs(save_dir, exist_ok=True)

        model.resize_token_embeddings(len(tokenizer))
        training_args = TrainingArguments(num_train_epochs=num_epochs,output_dir=save_dir,
                                          per_device_train_batch_size=batch_size,
                                          learning_rate=initial_lr,
                                          load_best_model_at_end=True,
                                          evaluation_strategy=IntervalStrategy.EPOCH,
                                          metric_for_best_model='eval_accuracy',
                                          save_total_limit=patience+1,
                                          save_strategy = IntervalStrategy.EPOCH,
                                          # # prediction_loss_only=True,
                                          gradient_accumulation_steps=int(32/batch_size), eval_accumulation_steps=int(32/batch_size),
                                          # fp16=True, fp16_full_eval=True,
                                          per_device_eval_batch_size=batch_size,
                                          # # debug="underflow_overflow"
                                          )

        param_names = []
        for name, param in model.named_parameters():
            param_names.append(name)

        # positional_embedding_params = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight',
        #      'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.LayerNorm.weight',
        #      'bert.embeddings.LayerNorm.bias']
        positional_embedding_params = ['bert.embeddings.position_embeddings.weight']
        non_positional_embedding_params = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight']
        pooler_params = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
        classifier_params = ['intermediate_classifier.weight', 'intermediate_classifier.bias', 'classifier.weight', 'classifier.bias']
        layer_norm_params = []
        attention_params = []
        for l in param_names:
            if 'LayerNorm' in l:
                layer_norm_params.append(l)
            elif l not in positional_embedding_params+non_positional_embedding_params+pooler_params+classifier_params:
                attention_params.append(l)
        print(len(positional_embedding_params+non_positional_embedding_params+layer_norm_params+attention_params+pooler_params+classifier_params), len(param_names))
        unfrozen_params = []
        unfrozen_params += classifier_params
        if freeze_positional == 0:
            unfrozen_params += positional_embedding_params
        if freeze_non_positional == 0:
            unfrozen_params += non_positional_embedding_params
        if freeze_layer_norm == 0:
            unfrozen_params += layer_norm_params
        if freeze_pooler == 0:
            unfrozen_params += pooler_params
        if freeze_attention == 0:
            unfrozen_params += attention_params

        frozen_counter = 0
        grad_counter = 0
        for name, param in model.named_parameters():
            if name in unfrozen_params:
                param.requires_grad = True
                grad_counter += len(param.flatten())
            else:
                param.requires_grad = False
                frozen_counter += len(param.flatten())

        print('Frozen parameters:', frozen_counter, grad_counter, grad_counter+frozen_counter, grad_counter*100/(grad_counter+frozen_counter))
        callbacks = []
        if early_stopping == 1:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

        model_path = "./EC1_STS_models/" + EC1_STS_models_list[EC1_STS_models]
        temp_extension = glob.glob(model_path+'/*')
        print("EC1_STS_model: ", model_path)
        print("Fractions: ", Fractions[EC1_STS_models])
        # config = AutoConfig.from_pretrained("./EC1_STS_models/models_STS_EC1_0/1022-092131")
        config = AutoConfig.from_pretrained(temp_extension[0])
        model = MyBertForSequenceClassification.from_pretrained(temp_extension[0], config=config)


        trainer = None
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets_train,
            eval_dataset=tokenized_datasets_test,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            data_collator=data_collator,
            tokenizer = tokenizer
        )

        # Load checkpoint model:
        # checkpoint = torch.load("./models_random_EC1_EX/0805-135007")


        # trainer.train()

        # dirs = [x[0] for x in os.walk(save_dir)]
        # for d in dirs:
        #     if 'checkpoint' in d:
        #         shutil.rmtree(d, ignore_errors='True')
        # trainer.save_model()

        # print(trainer.evaluate())
        # Loading saved model:
        # config = AutoConfig.from_pretrained("./models_random_EC1_EX/0805-135007")
        # model = MyBertForSequenceClassification.from_pretrained("./models_random_EC1_EX/0805-135007", config=config)


        # all_predictions_train_EC1 = trainer.predict(test_dataset=tokenized_datasets_train)
        # all_labels_train_EC1 = np.argmax(np.array(all_predictions_train_EC1[1]), axis=-1)
        # all_logits_train_EC1 = all_predictions_train_EC1[0]
        # all_predictions_train_EC1 = np.argmax(np.array(all_predictions_train_EC1[0]), axis=-1)
        # accuracy_train_EC1 = (np.sum(all_predictions_train_EC1 == all_labels_train_EC1))/len(all_labels_train_EC1)

        all_predictions_eval_EC1 = trainer.predict(test_dataset=tokenized_datasets_test)
        all_labels_eval_EC1 = np.argmax(np.array(all_predictions_eval_EC1[1]), axis=-1)
        all_logits_eval_EC1 = all_predictions_eval_EC1[0]
        all_predictions_eval_EC1 = np.argmax(np.array(all_predictions_eval_EC1[0]), axis=-1)
        accuracy_eval_EC1 = (np.sum(all_predictions_eval_EC1 == all_labels_eval_EC1)) / len(all_labels_eval_EC1)

        print("Accuracy: ", accuracy_eval_EC1)
        accuracy_matrics_Random_EC1_STS = [Fractions[EC1_STS_models], all_predictions_eval_EC1, all_labels_eval_EC1, all_logits_eval_EC1,
                                           accuracy_eval_EC1]

        accuracy_matrics_Random_EC1_STS_list.append(accuracy_matrics_Random_EC1_STS)


with open('./Accuracy_matrics_random_EC1_STS.pkl', 'wb') as f:
    pickle.dump(accuracy_matrics_Random_EC1_STS_list, f)

