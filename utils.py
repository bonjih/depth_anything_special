import os
import pandas as pd

import numpy as np
import cv2

import global_params_variables

params = global_params_variables.ParamsDict()

output_path_vid = params.get_value('output_video_path')



def make_csv(entry):
    with open('audit_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['ROI', 'TS(s)', 'Mean', 'Trend']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        roi, ts, mean, trend = entry
        print(roi, ts, mean, trend )
        writer.writerow({'ROI': roi, 'TS(s)': ts, 'Mean': mean, 'Trend': trend})


def make_csv2(data):
    if data is not None:
        os.makedirs(os.path.dirname(output_path_vid), exist_ok=True)
        file_empty = not os.path.exists(output_path_vid) or os.stat(output_path_vid).st_size == 0
        #print(data)
        # Concatenate DataFrames and filter out rows where 'roi' is 'None'
        # final_df = pd.concat([df[df['roi'] != 'None'] for df in data], ignore_index=True)
        # print(final_df)
        # Save final_df to CSV
        #final_df.to_csv('output.csv', index=False)

        # data_out = pd.DataFrame({'time_roi_1': [data[0][0]], 'sum_features_roi_1': [data[0][1]],
        #                          'result_roi_1': [data[0][2]], 'win_roi_1': [data[0][3]], 'psd_val_1': [data[0][4]],
        #                          'time_roi_2': [data[1][0]], 'sum_features_roi_2': [data[1][1]],
        #                          'result_roi_2': [data[1][2]], 'win_roi_2': [data[1][3]], 'psd_val_2': [data[1][4]]
        #                          })
        # if file_empty:
        #     data_out.to_csv(path, mode='w', index=False)
        # else:
        #     data_out.to_csv(path, mode='a', header=False, index=False)




def get_contours(gray, mask):
    diff_roi = cv2.bitwise_and(gray, gray, mask=mask)
    _, thresh = cv2.threshold(diff_roi, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def get_centre(cnts):
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY


def create_rectangle_array(cX, cY):
    # Calculate the starting and ending coordinates of the rectangle
    startX = max(0, cX - 50)
    endX = min(99, cX + 50)
    startY = max(0, cY - 50)
    endY = min(99, cY + 50)

    # Create the rectangle coordinates array
    rectangle_coordinates = np.array([[startX, startY],
                                      [endX, startY],
                                      [endX, endY],
                                      [startX, endY],
                                      [startX, startY]])

    # Create an empty array
    rectangle = np.zeros((100, 100))

    # Marking the rectangle area with 1s
    rectangle[startY:endY + 1, startX:endX + 1] = 1

    return rectangle_coordinates


def calculate_audit_means(audit_list):
    roi_data = {}

    for audit in audit_list:
        roi_key = audit[0]
        audit_2 = audit[2]
        audit_3 = audit[3]

        if roi_key not in roi_data:
            roi_data[roi_key] = {'audit_2_sum': 0, 'audit_3_sum': 0, 'count': 0}

        roi_data[roi_key]['audit_2_sum'] += audit_2
        roi_data[roi_key]['audit_3_sum'] += audit_3
        roi_data[roi_key]['count'] += 1

    roi_means = {}

    for roi_key, data in roi_data.items():
        if data['count'] > 0:
            audit_2_mean = data['audit_2_sum'] / data['count']
            audit_3_mean = data['audit_3_sum'] / data['count']
            roi_means[roi_key] = (audit_2_mean, audit_3_mean)

    return roi_means
