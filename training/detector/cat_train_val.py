import numpy as np
import os,csv
import pandas as pd
import random

path = '/data/public_data/tianchi2/chestCT_round1_annotation_new.csv'
csv_read = csv.reader(open(path,'r'))
names = []
for row in csv_read:
    if 'seriesuid' in row:
        continue
    if row[0] not in names:
        names.append(row[0])
print(len(names))

test_names = np.load('./test.npy')
test_names_ = []
for name in test_names:
    test_names_.append(name)

new_test_names = test_names_[:100]
print(len(new_test_names))

train_val_names = []
for name in names:
    if name not in new_test_names:
        train_val_names.append(name)

print(len(train_val_names))
np.save('train_val.npy', train_val_names)
np.save('new_test.npy', new_test_names)
#train_path = './train.npy'
#train_names = np.load(train_path)
#val_path = './val.npy'
#val_names = np.load(val_path)
#all_ct= []
#for name in train_names:
#    all_ct.append(name)
#print(len(all_ct))
#for name in val_names:
#    all_ct.append(name)
#print(len(all_ct))
#np.save('train_val.npy', all_ct)