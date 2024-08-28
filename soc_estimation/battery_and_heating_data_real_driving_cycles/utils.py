import os
import pandas as pd


def csvs2dfs(trips_dir, csvs_list):
    '''
    Getter list of dataframes from a list of .csv files.
    '''
    trips = []
    for i in range(len(csvs_list)):
        trip_name = csvs_list[i]
        trip_path = os.path.join(trips_dir, trip_name)
        trip = pd.read_csv(trip_path)
        trips.append(trip)
    return trips


def get_steady_cols(df):
    '''
    Getter steady columns from dataframe.
    '''
    steady_cols = [col for col in df.columns if df[col].nunique() == 1]
    return steady_cols


def data_cleaning(dfs, steady_cols_to_drop):
    min_SOC = dfs[0]._get_value(0, steady_cols_to_drop[0]) 
    max_SOC = dfs[0]._get_value(0, steady_cols_to_drop[1]) 
    cols_dfs = DataCleaning.get_cols_dfs(dfs)
    no_cols_dfs = DataCleaning.get_no_cols_dfs(dfs)
    min_cols_idx = DataCleaning.get_min_cols_idx(cols_dfs, no_cols_dfs)
    sup_min_cols_dfs_idx = DataCleaning.get_sup_min_cols_dfs_idx(cols_dfs, no_cols_dfs)
    set_min_cols = set(cols_dfs[min_cols_idx[0]])
    set_cols_to_delete = DataCleaning.get_set_cols_to_delete(cols_dfs, set_min_cols)
    DataCleaning.update_cols_dfs(dfs, sup_min_cols_dfs_idx, set_cols_to_delete)
    for i in range(len(dfs)):
        '''
        Steady columns dropping from dataframes.
        '''
        dfs[i].drop(steady_cols_to_drop, axis=1, inplace=True)
    dfs.pop(-1)
    return min_SOC, max_SOC


class DataCleaning:
    @staticmethod
    def get_cols_dfs(dfs):
        '''
        Getter each dataframe columns.
        '''
        cols_dfs = []
        for i in range(len(dfs)):
            cols_dfs.append(dfs[i].columns)
        return cols_dfs

    @staticmethod
    def get_no_cols_dfs(dfs):
        '''
        Getter number of each dataframe columns.
        '''
        no_cols_dfs = []
        for i in range(len(dfs)):
            no_cols_dfs.append(len(dfs[i].columns))
        return no_cols_dfs

    @staticmethod
    def get_min_cols_idx(cols_dfs, no_cols_dfs):
        '''
        Getter index of dataframes which have minimum number of columns.
        '''
        min_cols_idx = []
        for i in range(len(cols_dfs)):
            if len(cols_dfs[i]) == min(no_cols_dfs):
                min_cols_idx.append(i)
        return min_cols_idx

    @staticmethod
    def get_sup_min_cols_dfs_idx(cols_dfs, no_cols_dfs):
        '''
        Getter index of dataframes which have a number of columns greater than minimum number of columns.
        '''
        sup_min_cols_dfs_idx = []
        for i in range(len(cols_dfs)):
            if len(cols_dfs[i]) > min(no_cols_dfs):
                sup_min_cols_dfs_idx.append(i)
        return sup_min_cols_dfs_idx

    @staticmethod
    def get_set_cols_to_delete(cols_dfs, set_min_cols):
        '''
        Getter array of sets of columns to delete (each dataframe).
        '''
        set_cols_to_delete = []
        for i in range(len(cols_dfs)):
            curr_set_cols = set(cols_dfs[i])
            to_delete = curr_set_cols - set_min_cols
            set_cols_to_delete.append(to_delete)
        return set_cols_to_delete

    @staticmethod
    def update_cols_dfs(dfs, sup_min_cols_dfs_idx, set_cols_to_delete):
        '''
        Dataframes update. It provides for the columns dropping from dataframes.
        '''
        for i in range(len(sup_min_cols_dfs_idx)):
            # si prende l'indice corrispondente al dataframe con cols>min_cols
            idx = sup_min_cols_dfs_idx[i]
            dfs[idx].drop(list(set_cols_to_delete[idx]), axis=1, inplace=True)
            set_cols_to_delete[idx].clear()
