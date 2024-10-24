from typing import Dict, List
import re
import pandas as pd
import itertools


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    s=0
    r=[]
    while s < len(lst):
        subList = lst[s:s+n]

        # METHOD 1
        for i in range(len(subList)-1,-1,-1):
            r.append(subList[i])

        # METHOD 2
        #r += subList[::-1]

        s+=n
        
    return r


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    unique_count_list = []

    for i in lst:
        if len(i) not in unique_count_list:
            unique_count_list.append(len(i))

    dict = {}

    for i in sorted(unique_count_list):
        pair = {i : [j for j in lst if len(j) == i]}
        dict.update(pair)

    return dict

def flatten_dict(nested_dict: Dict,parent_key='', sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    dict = {}
    
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, Dict):
            dict.update(flatten_dict(v, new_key, sep=sep))
        
        elif isinstance(v, List):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, Dict):
                    dict.update(flatten_dict(item, list_key, sep=sep))
                else:
                    dict[list_key] = item
        
        else:
            dict[new_key] = v
    
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    perms = itertools.permutations(nums)
    
    unique_perms = set(perms) 

    return [list(p) for p in unique_perms]



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r'\b(?:(\d{2}-\d{2}-\d{4})|(\d{2}/\d{2}/\d{4})|(\d{4}\.\d{2}\.\d{2}))\b'
    matches = re.findall(date_pattern, text)

    # Method 1
    result = []
    for match in matches:
        date = ''.join(filter(lambda x: len(x)!=0, match))
        result.append(date)

    # Method 2
    # result = []
    # for match in matches:
    #     for date in match:
    #         if len(date)!=0:
    #             result.append(date)
    return result
    

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[i][j] = matrix[n - j - 1][i]
    
    final_matrix = [[0] * n for _ in range(n)] 
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()



reverse_list = reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8],3)

output_dict = group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"])


flat_dict = flatten_dict({
                            "road": {
                                "name": "Highway 1",
                                "length": 350,
                                "sections": [
                                    {
                                        "id": 1,
                                        "condition": {
                                            "pavement": "good",
                                            "traffic": "moderate"
                                        }
                                    }
                                ]
                            }
                        })

unique_perm = unique_permutations([1,1,2])

dates = find_all_dates("I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23.")

final_matrix = rotate_and_multiply_matrix([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

