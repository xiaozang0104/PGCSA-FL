import os

import numpy as np

from utils import load_image, dense_to_one_hot


class GetDataSet(object):
    def __init__(self, train_images_path, test_images_path='', is_iid=False):
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path

        self.IMG_WIDTH = 30
        self.IMG_HEIGHT = 30

    def load_data(self):
        import csv
        base_dir = os.path.dirname(self.train_images_path)
        train_csv = os.path.join(base_dir, 'Train.csv')
        test_csv = os.path.join(base_dir, 'Test.csv')

        def _load(csv_path):
            imgs, lbls = [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = os.path.join(base_dir, row['Path'])
                    imgs.append(np.array(load_image(p, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))))
                    lbls.append(dense_to_one_hot(int(row['ClassId']), num_classes=43))
            return np.array(imgs), np.array(lbls)

        x_train, y_train = _load(train_csv)
        x_test, y_test = _load(test_csv)

        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_train = np.multiply(x_train, 1.0 / 255.0)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.multiply(x_test, 1.0 / 255.0)

        return x_train, x_test, y_train, y_test