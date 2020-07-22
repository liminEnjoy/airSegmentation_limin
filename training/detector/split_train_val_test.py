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
random.shuffle(names)
num = len(names)
train_ct = names[:int(num*0.7)]
val_ct = names[int(num*0.7):int(num*0.8)]
test_ct = names[int(num*0.8):]
np.save('train.npy', train_ct)
np.save('val.npy', val_ct)
np.save('test.npy', test_ct)