# Prot_EC
## A Transformer Based Deep Learning System for Accurate Annotation of Enzyme Commission Numbers

In this project a transformer based Deep Learning model has been developed that could predict the Enzyme commission Numbers of Enzymes from full scale sequences. The input
of the model are enzyme protein sequences and the output are the four EC numbers.

The model consists of four modified ProtBert modules which has been slectively finetuned to achieve state of the art accuracy on EC numbers. The model has been trained 
on the Uniprot Swissprot reviewed dataset on two types of splits: The first one is called the random split, which randomly divides the data into train, validation and testing sets.
Whereas, the second type is called clustered split, where the data has been split using UNIREF to make sure that the training and testing splits consists of different distributions
of sequences.

The model has been compared with Proteinfer by Goodle and performs better in terms of accuracy and F1 scores. Moreover, the model is able to retain very high accuracy even when
the training dataset is shrunk to 10% of its original size making it suitable in applications with very low amount of data. Fruthermore, the model accuracy is independant of
sequence length so it is able to preform with very long or short sequences. Lastly, the model tunes most of its hyperparameters by itself so it is easy to use and does not
require a separate validation set in order to train.

Dataset link: https://drive.google.com/file/d/1bZD67DqXv9LkYo0HCCEXW4USjgjgqBAY/view?usp=sharing
Trained models link: https://drive.google.com/file/d/1ObwqMIGE6A-gjr3lOTjaAWDhP0kbsJjL/view?usp=sharing

## Running Instructions:
Prot_EC could be used to predict EC numbers on custom sequences by carring out the following steps:

step1: Clone the repository in the local machine

step2: Download the models from the following directory and paste it into the project root:
https://drive.google.com/file/d/1ObwqMIGE6A-gjr3lOTjaAWDhP0kbsJjL/view?usp=sharing

step3: Install pytorch by running the following in the terminal:

>> pip install torch==1.10.2+cpu torchvision==0.11.3+cpu --extra-index-url https://download.pytorch.org/whl/cpu

step4: Install other necessary libraries by running the following in the terminal:

>> pip install -r requirements.txt

step5: Run evaluation on custom sequences: (Here "input_file" contains the user defined sequences and the correspoding EC numbers are saved in the "output_file")

>> python3 test_eval.py --input_file=test_eval_data.txt --output_file=output.txt
