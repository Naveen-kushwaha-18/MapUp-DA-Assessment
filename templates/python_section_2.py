import pandas as pd
import numpy as np
from datetime import time




def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    df = pd.read_csv(file_path)
    toll_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    toll_ids.sort()
    distance_matrix = pd.DataFrame(0.0, index=toll_ids, columns=toll_ids)
    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance
        for k in toll_ids:
            for i in toll_ids:
                for j in toll_ids:
                    if distance_matrix.at[i, j] == 0 and i != j:
                        distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                        return distance_matrix
    

    ##return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
     unrolled_data = []
for id_start in distance_matrix.index:
    for id_end in distance_matrix.columns:
        if id_start != id_end:
            unrolled_data.append([id_start, id_end, distance_matrix.at[id_start, id_end]])
            unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
            return unrolled_df

    ##return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
ref_avg_distance = unrolled_df[unrolled_df['id_start'] == reference_id]['distance'].mean()
lower_bound = ref_avg_distance * 0.9
upper_bound = ref_avg_distance * 1.1
filtered_ids = unrolled_df.groupby('id_start')['distance'].mean()
ids_within_threshold = filtered_ids[(filtered_ids >= lower_bound) & (filtered_ids <= upper_bound)].index.tolist()
return sorted(ids_within_threshold)

    ##return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
for vehicle, rate in rate_coefficients.items():
    unrolled_df[vehicle] = unrolled_df['distance'] * rate
return unrolled_df

    ##return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    start_days = []
    start_times = []
    end_days = []
    end_times = []
    for index, row in unrolled_df.iterrows():
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            start_days.append(day)
            end_days.append(day)
            start_times.append(time(0, 0, 0))
            end_times.append(time(23, 59, 59))
            if day in ['Saturday', 'Sunday']:
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    unrolled_df.at[index, vehicle] *= 0.7
            else:
                if time(0, 0, 0) <= row['start_time'] <= time(10, 0, 0):
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        unrolled_df.at[index, vehicle] *= 0.8
                elif time(10, 0, 0) <= row['start_time'] <= time(18, 0, 0):
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        unrolled_df.at[index, vehicle] *= 1.2
                else:
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        unrolled_df.at[index, vehicle] *= 0.8
                        unrolled_df['start_day'] = start_days
                        unrolled_df['end_day'] = end_days
                        unrolled_df['start_time'] = start_times
                        unrolled_df['end_time'] = end_times
                        return unrolled_df

    ##return df
