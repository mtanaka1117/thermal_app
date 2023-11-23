import pandas as pd
import datetime

csv_path = 'dummy.csv'
df_dict = {}
label_dict = {}

with open(csv_path) as f:
    reader = pd.read_csv(f, chunksize=1, names=["time", "place", "label", "bbox"])
    for i, df in enumerate(reader):
        label = df.at[i, "label"].astype(int)
        
        #ラベルが無いなら追加
        if label not in label_dict.keys():
            df_dict[label] = df
            df_time = pd.to_datetime(df.at[i,'time'], format='%Y-%m-%d %H:%M:%S.%f')
            label_dict[label] = [1, [float(c) for c in df.at[i,"bbox"][1:-1].split(',')], df_time]
        
        #ラベルが存在するなら
        else:
            label_dict[label][0] += 1
            df_bbox = [float(c) for c in df.at[i,"bbox"][1:-1].split(',')]
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