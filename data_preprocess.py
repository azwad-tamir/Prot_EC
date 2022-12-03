import numpy as np
import re
import io
import pandas as pd

ID_all = []
KW_all = []
SQ_all = []
SQ_head_all = []
sample_counter = 0
input_pattern = r'INPUT\((.+)\)'
ID_pattern = r'^ID   (.+)'
SQ_pattern = r'^SQ   (.+)'
KW_pattern = r'^KW   (.+)'
sample_end_pattern = r'//'


with open("uniprot_sprot.dat") as infile:
    KW_sample = []
    SQ_sample = ""
    is_SQ = False
    for line in infile:
        if re.findall(SQ_pattern, line):
            SQ_head_all.extend(re.findall(SQ_pattern, line))
            is_SQ = True
        elif is_SQ:
            if re.findall(sample_end_pattern, line):
                is_SQ = False
                KW_temp = [temp.strip().replace('.', '') for temp in KW_sample]
                KW_temp = [i for i in KW_temp if i]
                KW_all.append(KW_temp)
                SQ_all.append(SQ_sample.replace(' ',''))
                KW_sample = []
                SQ_sample = ""
            else:
                SQ_sample = (SQ_sample + line).replace('\n','')
        elif re.findall(ID_pattern, line):
            ID_all.append((re.findall(ID_pattern, line)[0]).split()[0])
        elif re.findall(KW_pattern, line):
            KW_sample.extend(re.findall(KW_pattern, line)[0].split(';'))


data_dict = {'ID': ID_all, 'Keyword': KW_all, 'Sequence': SQ_all}
data_df = pd.DataFrame(data_dict)
# data_df.to_csv(r'uniprot_sprot.csv', index=False)
data_df.to_pickle("uniprot_sprot.pkl")


