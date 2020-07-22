import os,csv
path = 'chestCT_round1_annotation.csv'
save_path = 'chestCT_round1_annotation_new.csv'

csv_read = csv.reader(open(path,'r'))

csv_write = csv.writer(open(save_path,'w'))
csv_write.writerow(['seriesuid','coordX','coordY','coordZ','diameter_mm'])

for row in csv_read:
    if 'seriesuid' in row:
        continue
    label = int(row[-1])
    if label == 1:
        label=0
    elif label == 5:
        label=1
    elif label == 31:
        label=2
    elif label == 32:
        label=3
    diameter_mm = max([float(row[4]),float(row[5]),float(row[6])])
    csv_write.writerow([row[0],row[1],row[2],row[3],diameter_mm])
	