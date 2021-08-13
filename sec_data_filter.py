from datetime import datetime as dt
import cv2, csv, os, glob
import numpy as np
import pandas as pd
import tensorflow.keras as k
import plotly.express as px

Nan = np.nan
idle = 0
on = 1
off = 2
for file in glob.glob("*.csv"):
    df = pd.read_csv(file)
    basename = os.path.basename(file)

    light_data = df["light_data"]

# make average column
    av = 0
    ave = 0
    for row, now in light_data.iteritems():
        data = (now + ave) / 2
        ave = data
        df.at[row, 'average'] = data
    average = df['average']

# # average filter
#     r1 = 0
#     n1 = 0
#     min_val = 9
#     offset = 2
#     threshold = 10
#     for row, n in average.iteritems():
#         r = light_data.loc[row]
#         if row > 0:
#             r1 = light_data.loc[row - 1]
#             n1 = average.loc[row - 1]
#         if abs(r - r1) < min_val:
#             df.loc[row, 'average_filter'] = idle
#             continue
#         if (r - r1) > (n - n1) + offset and (r - r1) - (n - n1) > threshold:
#             df.loc[row, 'average_filter'] = on
#         # if r > n*2 - offset:
#         # df.loc[row, 'average_filter'] = on
#         elif (r1 - r) > (n1 - n) + offset and (r1 - r) - (n1 - n) > threshold:
#             df.loc[row, 'average_filter'] = off
#         # elif r1 > n*2 - offset:
#         # df.loc[row, 'average_filter'] = off
#         else:
#             df.loc[row, 'average_filter'] = idle

# average filter
    r1 = 0
    n1 = 0
    min_val = 9
    offset = 2
    threshold = 7.5
    for row, n in light_data.iteritems():
        r = light_data.loc[row]
        if row > 0:
            r1 = light_data.loc[row - 1]
            n1 = average.loc[row - 1]
        if abs(r - r1) < min_val:
            df.loc[row, 'label'] = idle
            continue
        if (r - r1) > (n - n1) + offset and (r - r1) - (n - n1) > threshold:
            df.loc[row, 'label'] = on
        elif (r1 - r) > (n1 - n) + offset and (r1 - r) - (n1 - n) > threshold:
            df.loc[row, 'label'] = off
        else:
            df.loc[row, 'label'] = idle

# plot
    fig = px.line(df, x=['date_time'], y=['light_data', 'label', 'average'])
    fig.update_xaxes(rangeslider_visible=True)
    fig.show(renderer='browser')

# save to csv
    df.to_csv(f"test_result/{basename}", index=False)

