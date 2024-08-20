import os


class Config:
    
    def __init__(
        self, 
        data_dir, 
        train_data_subdir, 
        val_data_subdir, 
        test_data_subdir
    ):
        self.data_dir = data_dir
        print(f'data_dir = {os.listdir(data_dir)}')
        
        self.train_data_dir = f'{self.data_dir}/{train_data_subdir}'
        print(f'train_data_dir = {os.listdir(self.train_data_dir)}')
        
        self.val_data_dir = f'{self.data_dir}/{val_data_subdir}'
        print(f'val_data_dir = {os.listdir(self.val_data_dir)}')
        
        self.test_data_dir = f'{self.data_dir}/{test_data_subdir}'
        print(f'test_data_dir = {os.listdir(self.test_data_dir)}')
        

    def get_train_data_dir(self):
        return self.train_data_dir

    
    def get_val_data_dir(self):
        return self.val_data_dir

    
    def get_test_data_dir(self):
        return self.test_data_dir
