from datetime import datetime as dt
import cv2, csv, os, glob
import pandas as pd


light_col = "조도"
date_col = "측정일시 koKR"
idle = 0
on = 1
off = 2
path = '/home/z/data/test1/'
for file in glob.glob(path + "*.csv"):
    df = pd.read_csv(file)
    basename = os.path.basename(file)

    df = df[[light_col, date_col]]
    light_data = df[light_col]
    time_data = df[date_col]

    df = df.assign(average = df[light_col])
    df = df.assign(average_filter = df[light_col])
    average_filter = df['average_filter']

 # make average column
    av = 0
    ave = 0
    for row, now in light_data.iteritems():
        data = (now + ave) / 2
        ave = data
        df.at[row, 'average'] = data
    average = df['average']

# average filter
    r1 = 0
    n1 = 0
    min_val = 8
    offset = 5
    for row, n in average.iteritems():
        r = light_data.loc[row]
        if row > 0:
            r1 = light_data.loc[row-1]
            n1 = average.loc[row-1]
        if abs(r - r1) < min_val:
            df.loc[row, 'average_filter'] = idle
            continue
        if r > n*2 - offset:
            df.loc[row, 'average_filter'] = on
        elif r1 > n*2 - offset:
            df.loc[row, 'average_filter'] = off
        else:
            df.loc[row, 'average_filter'] = idle

# # plot
#     fig = px.line(df, x=date_col, y=[light_col, 'average_filter', 'average'])
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show(renderer='browser')

# save to csv
    df.to_csv(f"test_result/{basename}", index=False)
