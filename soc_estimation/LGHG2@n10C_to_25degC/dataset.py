import numpy as np
import pandas as pd
import utils
import logging
import os



class LGHG2():
    
    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._train_data_subdir = 'train'
        self._val_data_subdir = 'val'
        self._test_data_subdir = 'test'
        self._train_data_dir = os.path.join(self._data_dir, self._train_data_subdir)
        self._val_data_dir = os.path.join(self._data_dir, self._val_data_subdir)
        self._test_data_dir = os.path.join(self._data_dir, self._test_data_subdir)
        self._logger = logging.getLogger()
        self._logger.info(f'{self._data_dir} = {os.listdir(self._data_dir)}')
        self._logger.info(f'{self._train_data_dir} = {os.listdir(self._train_data_dir)}')
        self._logger.info(f'{self._val_data_dir} = {os.listdir(self._val_data_dir)}')
        self._logger.info(f'{self._test_data_dir} = {os.listdir(self._test_data_dir)}')
        
    
    def get_data_dir(self):
        return self._data_dir
    
    def get_train_data_dir(self):
        return self._train_data_dir
    
    def get_val_data_dir(self):
        return self._val_data_dir
    
    def get_test_data_dir(self):
        return self._test_data_dir   

    def get_dataset(self):
        return self._get_data()

    
    def _get_data(self):
        self.train_data_df = self._get_df(data_dir=self._train_data_dir, data_ext='.csv')
        self.val_data_df = self._get_df(data_dir=self._val_data_dir, data_ext='.csv')
        self.test_data_df = self._get_df(data_dir=self._test_data_dir, data_ext='.csv')

        self._check_df(df=self.train_data_df, df_name='train_data_df')
        self._check_df(df=self.val_data_df, df_name='val_data_df')
        self._check_df(df=self.test_data_df, df_name='test_data_df')

        self._logger.info(f'data_columns = {self.train_data_df.columns}')
        cols = set(self.train_data_df.columns)
        _target = ['SOC']
        self._logger.info(f'target = {_target}')
        _target = set(_target)
        _features = cols - _target
        _target = list(_target)
        _features = list(_features)
        self._logger.info(f'features = {_features}')

        self._write_cols(dir_path=self._data_dir, file_name='target.txt', cols=_target)
        self._write_cols(dir_path=self._data_dir, file_name='features.txt', cols=_features)

        self._X_train = self.train_data_df[_features].values
        self._X_val = self.val_data_df[_features].values
        self._X_test = self.test_data_df[_features].values

        self._y_train = self.train_data_df[_target].values
        self._y_val = self.val_data_df[_target].values
        self._y_test = self.test_data_df[_target].values

        return (
            utils.normalize(self._X_train), 
            utils.normalize(self._X_val), 
            utils.normalize(self._X_test), 
            self._y_train, 
            self._y_val, 
            self._y_test
        )


    def get_X_train(self):
        return self._X_train
        
    def get_X_val(self):
        return self._X_val
        
    def get_X_test(self):
        return self._X_test   
        
    def get_y_train(self):
        return self._y_train   
        
    def get_y_val(self):
        return self._y_val 
        
    def get_y_test(self):
        return self._y_test 

    
    def _get_df(self, data_dir, data_ext):
        listdir = os.listdir(data_dir)
        dfs = []
        for data_name in listdir:
            if data_name.endswith(data_ext):
                data_path = os.path.join(data_dir, data_name)
                data_df = pd.read_csv(data_path)
                dfs.append(data_df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    
    def _check_df(self, df, df_name):
        if df.empty:
            self._logger.info(f'WARNING# {df_name} is empty.')
        else: 
            self._logger.info(f'{df_name} loaded.')

    
    def _write_cols(self, dir_path, file_name, cols):
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'w') as file:
            for i in range(len(cols)):
                file.write(f'{cols[i]}\n')
    