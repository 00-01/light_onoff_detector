from datetime import datetime
import glob, re
import pandas as pd
import numpy as np
import plotly.express as px


filename = 0
path = "/home/z/Desktop/"
for name in glob.glob(path + "*.log"):
    # column_names = ["date_time", "light_data", "light_data_average", "label"]
    df = pd.DataFrame()#columns=column_names)
    with open(name, 'r') as r:
        time_col = []
        data_col = []
        label_col = []
        line = r.readlines()
        n = len(line)
        for i in range(n):
            sec = 0
            # dt = line[i][1:18]
            dt = line[i][1:19]
            temp_sec = line[i][18:20]
            second = int(temp_sec)+sec
            light = line[i][27:114].replace("  ", "").split(",")
            for j in light:
                if len(j) > 6:
                    continue
                try :
                    float(j)
                    time_col.append(dt+str(sec))
                    data_col.append(int(j[:2]))
                    sec += 1
                except :
                    pass
            sec = 0
    df["date_time"] = time_col
    df["light_data"] = data_col

    # save to csv
    df.to_csv(f'{filename}.csv', index=False)
    filename += 1


















    # # 5. Chucking
    # line[0].split(']')
    # line[0].split(']')[1]
    # line[0].split(']')[1].upper()
    #
    # # 6. Datetime
    # s = line[0].split(']')[0].strip('[')
    # dtfmt ='%m/%d/%Y %I:%M:%S %p'   # %H -> 24 hours, %I-> 12 hours, for ISO 8601 format, use: %Y-%m-%dT%H:%M:%S.%f%z
    #
    # dt = datetime.strptime(s, dtfmt)
    #
    # # 7. Put into a data frame
    # # first, we need to determine columns
    # col1 = []
    # col2 = []
    # col3 = []
    #
    # # then, we fill the columns
    # for l in line:
    #     s1 = l.split(']')[0].strip('[')
    #     dt = datetime.strptime(s1, dtfmt)
    #     col1.append(dt)
    #     s = l.split(']')[1].strip().split(':')
    #     col2.append(s[0])
    #     if len(s) == 2:
    #         col3.append(s[1])
    #     else:
    #         col3.append(np.nan)
    #
    # # finally, we create the dataframe
    # df = pd.DataFrame([col1,col2,col3])
    # df = df.T
    # df.columns=['datetime','event_name', 'event_result']
    #
    # # 8. Normalize datetime to seconds
    # df['delta_t'] =df.datetime - df.datetime[0]
    #
    # # convert that to seconds
    # df['delta_t_seconds'] = 0
    # for i in range(df.shape[0]):
    #     df.ix[i,'delta_t_seconds'] = df.delta_t.iloc[i].seconds
    #
    # # 9. Save to csv file
