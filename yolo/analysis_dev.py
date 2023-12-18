import datetime
import csv

csv_path = 'example.csv'
# df_dict = {}
label_dict = {}

with open(csv_path) as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[2]
        
        if label not in label_dict.keys():
            df_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
            label_dict[label] = [df_time, df_time, 1, [float(c) for c in row[3][1:-1].split(',')]]
            
        else:
            label_dict[label][2] += 1
            bbox = [float(c) for c in row[3][1:-1].split(',')]
            bbox_avg = label_dict[label][3]
            n = label_dict[label][2]
            
            #bboxの平均を更新
            label_dict[label][3] = [round((n*y+x)/(n+1), 4) for x, y in zip(bbox, bbox_avg)]
            
            #最後に物体が確認された時間
            last_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
    
            if last_time - label_dict[label][1] < datetime.timedelta(seconds=30):
                label_dict[label][1] = last_time

            #物体が一定時間を超えて再び確認された場合
            else:
                with open('results.csv', 'a', newline='') as w:
                    writer = csv.writer(w)
                    line = [label] + label_dict[label]
                    writer.writerow(line)
                
                label_dict[label] = [last_time, last_time, 1, [float(c) for c in row[3][1:-1].split(',')]]

    with open('results.csv', 'a', newline="") as w:
        writer = csv.writer(w)
        for label in label_dict.keys():
            line = [label] + label_dict[label]
            writer.writerow(line)
    

# with open(csv_path) as f:
#     reader = pd.read_csv(f, chunksize=1, names=["time", "place", "label", "bbox"])
#     for i, df in enumerate(reader):
#         label = df.at[i, "label"].astype(int)
        
#         #ラベルが無いなら追加
#         if label not in label_dict.keys():
#             df_dict[label] = df
#             df_time = pd.to_datetime(df.at[i,'time'], format='%Y-%m-%d %H:%M:%S.%f')
#             label_dict[label] = [1, [float(c) for c in df.at[i,"bbox"][1:-1].split(',')], df_time]
        
#         #ラベルが存在するなら
#         else:
#             label_dict[label][0] += 1
#             df_bbox = [float(c) for c in df.at[i,"bbox"][1:-1].split(',')]
#             bbox_avg = label_dict[label][1]
#             n = label_dict[label][0]
            
#             #bboxの平均を更新
#             label_dict[label][1] = [(n*y+x)/(n+1) for x, y in zip(df_bbox, bbox_avg)]
            
#             df_time = pd.to_datetime(df.at[i,'time'], format='%Y-%m-%d %H:%M:%S.%f')

#             if n == 2:
#                 df_dict[label] = pd.concat([df_dict[label], df])
            
#             elif df_time - label_dict[label][2] < datetime.timedelta(minutes=1):
#                 df_dict[label] = df_dict[label][:-1]
#                 df_dict[label] = pd.concat([df_dict[label], df])

#             else:
#                 label_dict[label][0] = 1
#                 df_dict[label] = pd.concat([df_dict[label], df])
            
        
#     # print(df_dict)
#     for df in df_dict.values():
#         df.to_csv('result.csv',columns=["time", "place", "label", "bbox"], index=False, header=False, mode="a")
