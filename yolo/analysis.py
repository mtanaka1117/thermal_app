import pandas as pd
import datetime
import math

def calc_dist(bbox1, bbox2):
    dist1 = math.sqrt((bbox1[0]-bbox2[0])**2 + (bbox1[1]-bbox2[1])**2)
    dist2 = math.sqrt((bbox1[2]-bbox2[2])**2 + (bbox1[3]-bbox2[3])**2)
    return dist1+dist2

with open('dummy.csv') as f:
    df = pd.read_csv(f, names = ["time", "place", "label", "bbox", "bb_dist"])
    df_s = df.sort_values(['bbox']) #bbox順に並び替え
    
    df_s['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f') #str -> datetime型に変換
    df_s['bbox'] = df_s.bbox.apply(lambda x: list(map(float, x[1:-1].split(',')))) #bboxをstr -> floatのリストに変換

    df_s = df_s.reset_index() #新しいindexを割り振る
    
    #bboxの距離をbb_distに格納（bbox順に並び替えているので別の物体のbboxだと外れ値になるはず）
    bbox = df_s['bbox']
    df_s.at[0, 'bb_dist'] = 0
    for i in range(1, len(bbox)):
        bb_dist = calc_dist(bbox[i-1], bbox[i])
        if bb_dist < 10:
            df_s.at[i, 'bb_dist'] = bb_dist
        else:
            df_s.at[i, 'bb_dist'] = 0 #外れ値を0とする

    df_s['group_no'] = (df_s.bb_dist == 0).cumsum() #物体ごとにgroup_noを割り振る
    df_split = {n: df_s.iloc[rows].reset_index(drop=True) for n, rows in df_s.groupby('group_no').groups.items()} #物体ごとにdataframeを分割
    
    #分割したdataframeごとに、bbox順から元のindex順（時間順）に戻す
    for i in df_split.values():
        i.sort_values('index', inplace=True)
        i.drop('index', axis=1, inplace=True)
        i.reset_index(drop=True, inplace=True)
    
    #分割したdataframeから最頻値以外のラベルを取り除き結合
    firstLoop = True
    for i in df_split:
        mode_val = df_split[i]['label'].mode().values #ラベルの最頻値
        for j in range(len(df_split[i])):
            if df_split[i].at[j, 'label']!=mode_val:
                df_split[i].drop(j, inplace=True)
        
        if firstLoop: 
            df_con = df_split[i]
            firstLoop = False
        else:
            df_con = pd.concat([df_con, df_split[i]])
        
    df_con = df_con.reset_index(drop=True) #index振り直し
    label = df_con['label']
    time = df_con['time']
    thres_time = datetime.timedelta(minutes=1)
    
    #一定時間以内で、既にあるラベルは取り除く
    for i in range(1, len(time)-1):
        if label[i]==label[i-1] and time[i]-time[i-1]<thres_time and label[i]==label[i+1]:
            df_con.drop(i, inplace=True)
    
    #時間順に並び替え
    df_con = df_con.sort_values(['time']).reset_index(drop=True)
    df_con.to_csv('result.csv',columns=["time", "place", "label", "bbox"], index=False, header=False)
        
    