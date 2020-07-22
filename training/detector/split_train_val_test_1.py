import numpy as np
import os
import pandas as pd
import random
path = '/data/public_data/tianchi2/preprocess'
names = os.listdir(path)
all_ct=[]
for name in names:
    if name.endswith('_clean.npy'):
        all_ct.append(name.split('_')[0])
#         all_ct.append(bytes(name.split('_')[0], encoding="utf8"))
random.shuffle(all_ct)
num = len(all_ct)
train_ct = all_ct[:int(num*0.7)]
val_ct = all_ct[int(num*0.7):int(num*0.8)]
test_ct = all_ct[int(num*0.8):]
np.save('train.npy', train_ct)
np.save('val.npy', val_ct)
np.save('test.npy', test_ct)