from datetime import datetime as dt
import cv2, csv, os, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as k
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.math import confusion_matrix


def draw_CM(label, predicted):
    cm = confusion_matrix(label, predicted)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    return plt.show()

idle = 0
on = 1
off = 2
path = '/home/z/data/label/'
for file in glob.glob(path + "*.csv"):
    df = pd.read_csv(file)
    basename = os.path.basename(file)

    df = df[['device_field03', 'illuminance_onoff', 'device_data_reg_dtm']]
    light_data = df['device_field03']
    label_data = df['illuminance_onoff']
    time_data = df['device_data_reg_dtm']

    df = df.assign(average = df['device_field03'])
    df = df.assign(average_filter = df['illuminance_onoff'])
    average_filter = df['average_filter']

# relabel to 0, 1, 2 = idel, on, off
    prev = 0
    for row, label in label_data.iteritems():
        if prev == 0 and label == 1:
            df.loc[row, 'illuminance_onoff'] = on
        elif prev == 1 and label == 0:
            df.loc[row, 'illuminance_onoff'] = off
        else:
            df.loc[row, 'illuminance_onoff'] = idle
        prev = label

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
    min_val = 9
    offset = 2
    threshold = 7.5
    for row, n in average.iteritems():
        r = light_data.loc[row]
        if row > 0:
            r1 = light_data.loc[row - 1]
            n1 = average.loc[row - 1]
        if abs(r - r1) < min_val:
            df.loc[row, 'average_filter'] = idle
            continue
        if (r - r1) > (n - n1) + offset and (r - r1) - (n - n1) > threshold:
            df.loc[row, 'average_filter'] = on
        elif (r1 - r) > (n1 - n) + offset and (r1 - r) - (n1 - n) > threshold:
            df.loc[row, 'average_filter'] = off

        else:
            df.loc[row, 'average_filter'] = idle

# # average filter
#     r1 = 0
#     n1 = 0
#     min_val = 9
#     offset = 5
#     for row, n in average.iteritems():
#         r = light_data.loc[row]
#         if row > 0:
#             r1 = light_data.loc[row-1]
#             n1 = average.loc[row-1]
#         if abs(r - r1) < min_val:
#             df.loc[row, 'average_filter'] = idle
#             continue
#         if r > n*2 - offset:
#             df.loc[row, 'average_filter'] = on
#         elif r1 > n*2 - offset:
#             df.loc[row, 'average_filter'] = off
#         else:
#             df.loc[row, 'average_filter'] = idle

# plot
    # fig = px.line(df, x='device_data_reg_dtm', y=['device_field03', 'illuminance_onoff', 'new', 'average', 'average_filter'])
    fig = px.line(df, x='device_data_reg_dtm', y=['device_field03', 'average_filter', 'average', 'illuminance_onoff'])
    fig.update_xaxes(rangeslider_visible=True)
    fig.show(renderer='browser')

# cm
    draw_CM(df['illuminance_onoff'], average_filter)

# save to csv
    # df.to_csv(f"test/{basename}", index=False)
