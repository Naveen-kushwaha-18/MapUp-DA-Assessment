import pandas as pd
import numpy as np

def calculate_distance_matrix(df)-> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    toll_locations = sorted(set(df['id_start']).union(set(df['id_end'])))

    dist_matrix = pd.DataFrame(np.inf, index=toll_locations, columns=toll_locations)

    np.fill_diagonal(dist_matrix.values, 0)

    for i, row in df.iterrows():
        dist_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        dist_matrix.loc[row['id_end'], row['id_start']] = row['distance']


    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                if dist_matrix.loc[k, i] != np.inf and dist_matrix.loc[k, j] != np.inf:
                    dist_matrix.loc[i, j] = min(dist_matrix.loc[i, j], dist_matrix.loc[i, k] + dist_matrix.loc[k, j])

    return dist_matrix


def unroll_distance_matrix(df)->pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_df = df.unstack().reset_index()

    unrolled_df.columns = ['id_end', 'id_start', 'distance']

    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    
    unrolled_df = unrolled_df.reset_index(drop=True)
    
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame:
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
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    
    threshold_df = avg_distances[(avg_distances['distance'] >= (avg_distance * 0.9)) & (avg_distances['distance'] <= (avg_distance * 1.1))]['id_start']

    return threshold_df



def calculate_toll_rate(df)->pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    vehicles_rate = {'moto':0.8,'car':1.2,'rv':1.5,'bus':2.5,'truck':3.6}


    for vehicle,rate in vehicles_rate.items():
        df[vehicle] = df['distance']*rate
    df = df.drop(columns='distance')
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df



df = pd.read_csv('.//datasets//dataset-2.csv')

matrix_df = calculate_distance_matrix(df)

unrolled_df = unroll_distance_matrix(matrix_df)

threshold_df = find_ids_within_ten_percentage_threshold(unrolled_df,1001400)

vehicle_rate_df = calculate_toll_rate(unrolled_df)
print(vehicle_rate_df)


