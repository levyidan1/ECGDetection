from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
from os import listdir
from os.path import isfile, join
import csv
import cv2
from PIL import Image

config = configparser.ConfigParser()
path_ = os.path.dirname(os.path.abspath(__file__))
config.read(f'{path_}/config.ini')


class Generic_Dataset(Dataset):
    # Abst class that all datasets will inherit from
    def __init__(self, to_cut_image=False, to_equalize_hist=False, to_use_transform=False, **kwargs):
        super().__init__(**kwargs)
        self.stat_only = False
        self.perspective_transformer = T.RandomPerspective(distortion_scale=0.05, p=0.5)
        self.in_w = 1650
        self.in_h = 880
        self.in_channels = 3
        self.to_cut_image = to_cut_image
        self.to_equalize_hist = to_equalize_hist
        self.to_use_transform = to_use_transform
        self.stats = [0, 0]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def Scale_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = self.in_w  # Like the cut of NY DB
        height = self.in_h
        dsize = (width, height)
        # cv2.imshow('image window',img)
        # cv2.waitKey(0)
        img = cv2.resize(img, dsize)
        img = np.transpose(img, (2, 0, 1))
        return img

    def set_statistics_only(self, stat_only=False):
        self.stat_only = stat_only

    def Normalize_image(self, img):
        img_out = np.zeros_like(img, dtype=float)
        d0 = np.max(img[0]) - np.min(img[0]) * 1.0
        d1 = np.max(img[1]) - np.min(img[1]) * 1.0
        d2 = np.max(img[2]) - np.min(img[2]) * 1.0
        if d0 == 0:
            d0 = 1.0
        if d1 == 0:
            d1 = 1.0
        if d2 == 0:
            d2 = 1.0
        img_out[0] = (img[0] - np.min(img[0])) / d0
        img_out[1] = (img[1] - np.min(img[1])) / d1
        img_out[2] = (img[2] - np.min(img[2])) / d2
        return img_out

    def plot(self, idx):
        item_to_show = self.__getitem__(idx)
        item_dims = np.shape(item_to_show[0])
        print(f'Showing item {idx},  size : {item_dims}')
        plt.imshow(np.transpose(item_to_show[0], (1, 2, 0)))
        plt.show()
        return

    def calc_stat(self):
        self.set_statistics_only(True)
        for i in range(self.__len__()):
            _, classification = self.__getitem__(i)
            if classification:
                self.stats[0] += 1
            else:
                self.stats[1] += 1
        self.set_statistics_only(False)


class Source_Dataset(Generic_Dataset):
    # Generic Source Dataset that all source datasets will inherit from
    def __init__(self, classification_category='Atrial fibrillation', stat_only=False, **kwargs):
        super().__init__(**kwargs)
        self.classification_category = classification_category
        self.stat_only = stat_only

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class NY_Dataset(Source_Dataset):
    # NY Dataset
    def __init__(self, to_cut_image=True, **kwargs):
        super().__init__(**kwargs)
        self.all_possible_categories = []
        self.database_path = config['Image_database_path']['Images_Path']
        self.files_in_database = os.listdir(self.database_path)
        self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]
        self.classification_path = config['Image_database_path']['Classification_Path']
        self.server_db_path = config['Server_storage_path']['Server_Path']
        if self.stat_only == False:
            self.classifications = self.Upload_Classifications()
        self.to_cut_image = to_cut_image

    def __len__(self):
        return len(self.files_in_database)

    def __getitem__(self, index):
        try:
            classification_ = self.classifications[self.files_in_database[index][:-4]]
            if self.stat_only != False:
                return 0, classification_
            img = self.NY_db_image_upload(index)
            if self.to_equalize_hist:
                img = self.Normalize_image(img)
            return img, classification_
        except:
            print(f'Problematic index : {index}')

    def Upload_Classifications(self):
        classifications = {}
        Trues = 0
        Falses = 0
        with open(self.classification_path, newline='\n') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_idx, row in enumerate(reader):
                if row_idx > -1:
                    cl = [row[f'Dx{str(indx)}'] for indx in range(1, 11)]
                    for c in cl:
                        if c is not None:
                            self.all_possible_categories.append(c)
                    if not isinstance(self.classification_category, list):
                        classification = self.classification_category in cl
                    else:
                        E = [i for i in self.classification_category if i in cl]
                        classification = len(E) > 0
                    classifications[row['id']] = classification
                    if classification:
                        Trues += 1
                    else:
                        Falses += 1
            print(f'Trues : {Trues}, Falses: {Falses}')
            self.stats = [Trues, Falses]
        return classifications

    def NY_db_image_upload(self, idx):
        img = cv2.imread(os.path.join(self.database_path, self.files_in_database[idx]))
        # print(f'Uploaded: {self.files_in_database[idx]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.to_cut_image:
            img = img[270:1150, :, :]
        if self.to_use_transform:
            im = Image.fromarray(img)
            # im.save(f"Before_{idx}.jpg")
            img = self.perspective_transformer(im)
            img = np.array(img)
            # im = Image.fromarray(img)
            # im.save(f"After_{idx}.jpg")
        img = np.transpose(img, (2, 0, 1))
        return img


class Brazilian_Dataset(Source_Dataset):
    # Brazilian Dataset
    def __init__(self, classification_category, classification_path, database_path, **kwargs):
        super().__init__(classification_category=classification_category, **kwargs)
        self.classification_path = classification_path
        self.database_path = database_path
        self.files_in_database = os.listdir(self.database_path)
        self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]
        self.categories_lookup_dict = {'Atrial fibrillation': 'AF', 'AV Block': '1dAVb',
                                       'Bundle Branch Block - Right - RBBB': 'RBBB',
                                       'Bundle Branch Block - Left - LBBB': 'LBBB', 'Sinus bradycardia': 'SB'}
        self.valid_category = (
                self.classification_category in self.categories_lookup_dict
        )
        self.classifications = self.upload_classifications()

    def __len__(self):
        return len(self.files_in_database) if self.valid_category else 0

    def __getitem__(self, index):
        item = self.files_in_database[index]
        classification_ = self.classifications[item[:-4]]
        if self.stat_only != False:
            return 0, classification_
        img = cv2.imread(os.path.join(self.database_path, item))
        img = self.Scale_image(img)
        if self.to_equalize_hist:
            img = self.Normalize_image(img)
        return img, classification_

    def upload_classifications(self):
        if not self.valid_category:
            raise ValueError(
                f'Invalid classification category: {self.classification_category}'
            )
        classifications = {}
        num_true_classifications = 0
        num_false_classifications = 0
        with open(self.classification_path, newline='\n') as csvfile:
            reader = csv.DictReader(csvfile)
            # skip the first row:
            next(reader)
            for row in reader:
                classification = (row[self.categories_lookup_dict[self.classification_category]] == '1')
                classifications[row['filename']] = classification
                if classification:
                    num_true_classifications += 1
                else:
                    num_false_classifications += 1
            print(f'Trues : {num_true_classifications}, Falses: {num_false_classifications}')
            self.stats = [num_true_classifications, num_false_classifications]
        return classifications


class Combined_Dataset(Source_Dataset):
    # Combining all source datasets. Currently it's NY + Brazilian
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Br_ds = Brazilian_Dataset()
        self.NY_ds = NY_Dataset()
        self.valid_category = False
        if self.classification_category in self.Br_ds.categories_lookup_dict.keys():
            self.valid_category = True
        self.calc_stat()

    def __len__(self):
        if self.valid_category:
            return len(self.Br_ds) + len(self.NY_ds)
        else:
            return len(self.NY_ds)

    def __getitem__(self, index):
        if self.valid_category:
            if index < len(self.Br_ds):
                return self.Br_ds[index]
            else:
                return self.NY_ds[index - len(self.Br_ds)]
        else:
            return self.NY_ds[index]

    def set_statistics_only(self, stat_only=False):
        self.Br_ds.stat_only = stat_only
        self.NY_ds.stat_only = stat_only


class Mobile_Dataset(Generic_Dataset):
    # Providing images from mobile device
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Adversarial_path = config['Adversarial_path']['Adversarial_path']
        self.list_of_captured_from_mobile_files = [f for f in listdir(self.Adversarial_path) if
                                                   isfile(os.path.join(self.Adversarial_path, f))]

    def __len__(self):
        return len(self.list_of_captured_from_mobile_files)

    def __getitem__(self, index):
        if self.stat_only:
            return (0, True)
        else:
            img_filename = self.list_of_captured_from_mobile_files[index]
            img = cv2.imread(os.path.join(self.Adversarial_path, img_filename))
            img = self.Scale_image(img)
            if self.to_equalize_hist:
                img = self.Normalize_image(img)
            classification_ = True
            sample = (img, classification_)
        return sample


class Adversarial_Dataset(Generic_Dataset):
    # Half of images from mobile and half from labeled dataset
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_dataset = Combined_Dataset()
        self.mobile_dataset = Mobile_Dataset()

    def __len__(self):
        return 2 * min(len(self.source_dataset), len(self.mobile_dataset))

    def __getitem__(self, index):
        if index % 2 == 0:
            return self.source_dataset[index // 2]
        else:
            return self.mobile_dataset[(index - 1) // 2]

    def set_statistics_only(self, stat_only=False):
        self.source_dataset.set_statistics_only(stat_only=stat_only)
        self.mobile_dataset.set_statistics_only(stat_only=stat_only)


def Test_Specific_Dataset(Dataset, tag):
    print(f'Testing dataset : {tag}')
    ds = Dataset()
    item, classification = ds[0]
    # for i in range(len(ds)):
    #     _, classification_ = ds[i]
    print(f'Object of {tag} dataset created')
    print(f'Length of the dataset is: {len(ds)}')
    ds.set_statistics_only(True)
    for i in range(5):
        _, classification_ = ds[i]
        print(f'{i}th item shape is: {np.shape(_)}')
    print('All items are accessible statistically')
    ds.set_statistics_only(False)
    item_0, classification_ = ds[0]
    print(f'Item 0, shape : {np.shape(item_0)}, classification: {classification_}')
    item_1, classification_ = ds[1]
    print(f'Item 1, shape : {np.shape(item_1)}, classification: {classification_}')
    assert np.shape(item_0) == np.shape(item_1), 'Dimensions of subsequent frames are not the same'
    item_, classification_ = ds[len(ds) - 1]
    print(f'Item {len(ds) - 1}, shape : {np.shape(item_)}, classification: {classification_}')


def Test_database_constructor():
    ds = Combined_Dataset(classification_category='Sinus tachycardia', to_cut_image=True, to_equalize_hist=False,
                          to_use_transform=False)
    print('Finished testing of database constructor')


def Test_Datasets():
    print('Testing datasets...')
    Test_Specific_Dataset(Mobile_Dataset, 'Mobile_Dataset')
    Test_Specific_Dataset(Adversarial_Dataset, 'Adversarial_Dataset')
    Test_Specific_Dataset(Combined_Dataset, 'Combined_Dataset')
    Test_Specific_Dataset(Brazilian_Dataset, 'Brazilian_Dataset')
    Test_Specific_Dataset(NY_Dataset, 'NY_Dataset')
    print('Finished testing datasets...')


if __name__ == "__main__":
    Test_database_constructor()
    Test_Datasets()
