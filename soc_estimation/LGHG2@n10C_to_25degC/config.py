import os

class Config:
    """
    The `Config` class manages the directory paths for datasets used in 
    machine learning tasks. It initializes the paths to the training, 
    validation, and testing datasets based on the main directory and 
    subdirectory names provided.
    """

    def __init__(self, 
        data_dir:str, 
        train_data_subdir:str, 
        val_data_subdir:str, 
        test_data_subdir:str
    ):
        """
        Initializes the Config class with the directory structure.

        Parameters:
        - data_dir (str): The main directory that contains the dataset subdirectories.
        - train_data_subdir (str): The name of the subdirectory containing the training dataset.
        - val_data_subdir (str): The name of the subdirectory containing the validation dataset.
        - test_data_subdir (str): The name of the subdirectory containing the test dataset.

        The constructor also prints out the contents of each directory 
        to verify the presence of files or subdirectories.
        """
        self.data_dir = data_dir
        print(f'data_dir = {os.listdir(data_dir)}')

        self.train_data_dir = f'{self.data_dir}/{train_data_subdir}'
        print(f'train_data_dir = {os.listdir(self.train_data_dir)}')

        self.val_data_dir = f'{self.data_dir}/{val_data_subdir}'
        print(f'val_data_dir = {os.listdir(self.val_data_dir)}')

        self.test_data_dir = f'{self.data_dir}/{test_data_subdir}'
        print(f'test_data_dir = {os.listdir(self.test_data_dir)}')

    def get_train_data_dir(self) -> str:
        """
        Returns the path to the training data directory.

        Returns:
        - str: The full path to the training data directory.
        """
        return self.train_data_dir

    def get_val_data_dir(self) -> str:
        """
        Returns the path to the validation data directory.

        Returns:
        - str: The full path to the validation data directory.
        """
        return self.val_data_dir

    def get_test_data_dir(self) -> str:
        """
        Returns the path to the test data directory.

        Returns:
        - str: The full path to the test data directory.
        """
        return self.test_data_dir
