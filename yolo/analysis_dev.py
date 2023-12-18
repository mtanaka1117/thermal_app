import pandas as pd
import datetime
import math

def calc_dist(bbox1, bbox2):
    dist1 = math.sqrt((bbox1[0]-bbox2[0])**2 + (bbox1[1]-bbox2[1])**2)
    dist2 = math.sqrt((bbox1[2]-bbox2[2])**2 + (bbox1[3]-bbox2[3])**2)
    return dist1+dist2

csv_path = 'dummy.csv'
df_dict = {}
label_dict = {}

with open(csv_path) as f:
    reader = pd.read_csv(f, chunksize=1, names=["time", "place", "label", "bbox"])
    for i, df in enumerate(reader):
        label = df.at[i, "label"].astype(int)
        
        #ラベルが無いなら追加
        # df_dict.setdefault(label, df)
        if label not in label_dict.keys():
            df_dict[label] = df
            df_time = pd.to_datetime(df.at[i,'time'], format='%Y-%m-%d %H:%M:%S.%f')
            label_dict[label] = [1, [float(c) for c in df.at[i,"bbox"].split(',')], df_time]
            # print(df_bbox)
        
        #ラベルが存在するなら
        else:
            label_dict[label][0] += 1
            df_bbox = [float(c) for c in df.at[i,"bbox"].split(',')]
            bbox_avg = label_dict[label][1]
            n = label_dict[label][0]
            
            #bboxの平均を更新
            label_dict[label][1] = [(n*y+x)/(n+1) for x, y in zip(df_bbox, bbox_avg)]
            
            df_time = pd.to_datetime(df.at[i,'time'], format='%Y-%m-%d %H:%M:%S.%f')

            if n == 2:
                df_dict[label] = pd.concat([df_dict[label], df])
            
            elif df_time - label_dict[label][2] < datetime.timedelta(minutes=1):
                df_dict[label] = df_dict[label][:-1]
                df_dict[label] = pd.concat([df_dict[label], df])

            else:
                label_dict[label][0] = 1
                df_dict[label] = pd.concat([df_dict[label], df])
            
        
    # print(df_dict)
    for df in df_dict.values():
        df.to_csv('result.csv',columns=["time", "place", "label", "bbox"], index=False, header=False, mode="a")