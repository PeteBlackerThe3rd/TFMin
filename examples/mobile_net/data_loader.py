import numpy as np
import os
import matplotlib.pyplot as plt


class DataLoader:
    """Data Loader class. As a simple case, the model is tried on TinyImageNet. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.y_train = None
        self.img_mean = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size

    def load_data(self):

        path_of_script = os.path.dirname(os.path.realpath(__file__))

        # Please make sure to change this function to load your train/validation/test data.
        train_data = np.array([plt.imread(path_of_script + '/data/test_images/0.jpg'),
                               plt.imread(path_of_script + '/data/test_images/1.jpg'),
                               plt.imread(path_of_script + '/data/test_images/2.jpg'),
                               plt.imread(path_of_script + '/data/test_images/3.jpg')])
        self.X_train = train_data
        self.y_train = np.array([284, 264, 682, 2])

        val_data = np.array([plt.imread(path_of_script + '/data/test_images/0.jpg'),
                             plt.imread(path_of_script + '/data/test_images/1.jpg'),
                             plt.imread(path_of_script + '/data/test_images/2.jpg'),
                             plt.imread(path_of_script + '/data/test_images/3.jpg')])

        self.X_val = val_data
        self.y_val = np.array([284, 264, 682, 2])

        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        #img_height = 224
        #img_width = 224
        num_channels = 3
        return num_channels, self.train_data_len, self.val_data_len

    def generate_batch(self, type='train'):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if type == 'train':
            # Training time!
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        elif type == 'test':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                y_batch = self.y_test[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.test_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        elif type == 'val':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_val[start_idx:start_idx + self.batch_size]
                y_batch = self.y_val[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.val_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        else:
            raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")
