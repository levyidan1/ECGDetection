import itertools
import pandas as pd
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
import seaborn as sns
import wandb
# import pyvips


import categories

config = configparser.ConfigParser()
path_ = os.path.dirname(os.path.abspath(__file__))
config.read(path_ + '/config.ini')


class Generic_Dataset(Dataset):
    # Abst class that all datasets will inherit from
    def __init__(self, to_cut_image=False, to_equalize_hist=False, to_use_transform=False, **kwargs):
        super().__init__(**kwargs)
        self.stat_only = False
        self.perspective_transformer = T.RandomPerspective(distortion_scale=0.05, p=0.5,fill=255)
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

    def get_stats(self):
        return self.stats


class Source_Dataset(Generic_Dataset):
    # Generic Source Dataset that all source datasets will inherit from
    def __init__(self, classification_category=None, stat_only=False, **kwargs):
        super().__init__(**kwargs)
        self.classification_category = classification_category
        self.stat_only = stat_only
        self.database_valid_for_category = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def upload_classifications(self):
        raise NotImplementedError


class NY_Dataset(Source_Dataset):
    # NY Dataset
    def __init__(self, to_cut_image=True, classification_category=None, num_images_to_load=1000, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'NY'
        self.all_possible_categories = []
        if classification_category is not None:
            self.classification_category = classification_category
        self.database_path = config['Image_database_path']['Images_Path']
        self.files_in_database = os.listdir(self.database_path)
        self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]
        # self.files_in_database = self.files_in_database[:num_images_to_load]
        self.classification_path = config['Image_database_path']['Classification_Path']
        self.server_db_path = config['Server_storage_path']['Server_Path']
        if self.stat_only == False:
            self.classifications = self.upload_classifications()
        self.to_cut_image = to_cut_image
        self.to_use_cache = False
        self.img_cache = {}
        self.img_cache_size = 32768

    def __len__(self):
        if self.database_valid_for_category:
            return len(self.files_in_database)
        else:
            return 0

    def __getitem__(self, index):
        try:
            img_file = self.files_in_database[index]
            classification_ = self.classifications[img_file[:-4]]
            if self.stat_only:
                return (0, classification_)
            if self.to_use_cache:
                if img_file in self.img_cache:
                    img = self.img_cache[img_file]
                else:
                    img = self.NY_db_image_upload(index)
                    if len(self.img_cache) > self.img_cache_size:
                        self.img_cache.popitem(last=False)
                    self.img_cache[img_file] = img
            else:
                img = self.NY_db_image_upload(index)
            if self.to_equalize_hist:
                img = self.Normalize_image(img)
            return (img, classification_)
        except:
            print(f'Problematic index : {index}')

    def upload_classifications(self):
        classifications = {}
        classifications_hash = categories.dataset_lookup_dicts[self.dataset_name].get_categories_lookup_dict()
        Trues = 0
        Falses = 0
        if self.classification_category in classifications_hash.keys():
            self.database_valid_for_category = True
        if self.database_valid_for_category:
            with open(self.classification_path, newline='\n') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_idx, row in enumerate(reader):
                    if row_idx > -1:
                        cl = [row['Dx' + str(indx)] for indx in range(1, 11)]
                        for c in cl:
                            if c is not None:
                                self.all_possible_categories.append(c)
                        if isinstance(classifications_hash[self.classification_category], list) == False:
                            classification = classifications_hash[self.classification_category] in cl
                        else:
                            E = [i for i in classifications_hash[self.classification_category] if i in cl]
                            if len(E) > 0:
                                classification = True
                            else:
                                classification = False
                        # check if the image is in the database
                        filename = row['id'] + '.png'
                        if filename in self.files_in_database:
                            classifications[row['id']] = classification
                            if classification:
                                Trues += 1
                            else:
                                Falses += 1
        print(
            f'Database Name: NY, Classification Category: {self.classification_category}, Trues: {Trues}, Falses: {Falses}')
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

    def NY_db_image_upload_usingVIPSandShrink(self, index):
        f = os.path.join(self.database_path, self.files_in_database[index])
        image = pyvips.Image.new_from_file(f, access="sequential", shrink=4)
        image = image.colourspace("srgb")
        if self.to_cut_image:
            image = image.crop(270, 0, 1150, 1150)
        if self.to_use_transform:
            image = self.perspective_transformer(image)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        return image


class Brazilian_Dataset(Source_Dataset):
    # Brazilian Dataset
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(**kwargs)
        if classification_category is not None:
            self.classification_category = classification_category
        self.classification_path = config['Brazilian_dataset']['Classification_Path']
        self.database_path = config['Brazilian_dataset']['Images_Path']
        # if isinstance(ecg_format, str):
        #     self.database_path = self.database_path + '_format_' + ecg_format
        # else:
        #     self.database_path = self.database_path + '_format_' + str(ecg_format)
        self.files_in_database = os.listdir(self.database_path)
        self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]
        self.files_in_database = sorted(self.files_in_database, key=lambda x: int(x[4:-4]))
        self.dataset_name = 'Brazilian'
        self.categories_lookup_dict = categories.dataset_lookup_dicts[self.dataset_name].get_categories_lookup_dict()
        if self.classification_category in self.categories_lookup_dict:
            self.database_valid_for_category = True
        self.classifications = self.upload_classifications()

    def __len__(self):
        return len(self.files_in_database) if self.database_valid_for_category else 0

    def __getitem__(self, index):
        item = self.files_in_database[index]
        classification_ = self.classifications[item[:-4]]
        if self.stat_only == False:
            img = cv2.imread(os.path.join(self.database_path, item))
            img = self.Scale_image(img)
            if self.to_equalize_hist:
                img = self.Normalize_image(img)
            sample = (img, classification_)
        else:
            sample = (0, classification_)
        return sample

    def upload_classifications(self):
        classifications = {}
        Trues = 0
        Falses = 0
        if self.database_valid_for_category:
            classifications_file = pd.read_csv(self.classification_path)
            for index, row in classifications_file.iterrows():
                classification = False
                if isinstance(self.categories_lookup_dict[self.classification_category], list) == False:
                    if row[self.categories_lookup_dict[self.classification_category]] == 1:
                        classification = True
                else:
                    for item in self.categories_lookup_dict[self.classification_category]:
                        if row[item] == 1:
                            classification = True
                            break
                classifications[row['filename']] = classification
                if classification:
                    Trues += 1
                else:
                    Falses += 1
        print(
            f'Database Name: BR, Classification Category: {self.classification_category}, Trues: {Trues}, Falses: {Falses}')
        self.stats = [Trues, Falses]
        return classifications


class CategoriesMapping():
    def __init__(self):
        # self.categories_mapping_path = os.path.join(
        #     '/home/idanlevy/TSCA_App_Python/NY_Classifier/dx_mapping_unscored.csv')
        # self.categories_mapping_path_scored = os.path.join(
        #     '/home/idanlevy/TSCA_App_Python/NY_Classifier/dx_mapping_scored.csv')
        self.categories_mapping_path_full = os.path.join(
            '/home/idanlevy/TSCA_App_Python/NY_Classifier/dx_mapping.csv')
        self.categories_mapping = pd.read_csv(self.categories_mapping_path_full)
        # self.categories_mapping_scored = pd.read_csv(self.categories_mapping_path_scored)
        self.categories_full_name = self.categories_mapping['Dx'].values
        # self.categories_full_name_scored = self.categories_mapping_scored['Dx'].values
        self.categories_abbreviation = self.categories_mapping['Abbreviation'].values
        # self.categories_abbreviation_scored = self.categories_mapping_scored['Abbreviation'].values
        self.datasets = self.categories_mapping.keys()[3:-1]
        self.starting_index_for_dataset = {}
        # dataset names: ['CPSC', 'CPSC_Extra', 'StPetersburg', 'PTB', 'PTB_XL', 'Georgia',
        #        'Chapman_Shaoxing', 'Ningbo']
        # num recordings: cpsc_2018, 6,877 recordings
        # cpsc_2018_extra (China 12-Lead ECG Challenge Database – unused CPSC 2018 data), 3,453 recordings
        # st_petersburg_incart (12-lead Arrhythmia Database), 74 recordings
        # ptb (Diagnostic ECG Database,) 516 recordings
        # ptb-xl (electrocardiography Database), 21,837 recordings
        # georgia (12-Lead ECG Challenge Database), 10,344 recordings
        # chapman-shaoxing (Chapman University, Shaoxing People’s Hospital -12-lead ECG Database), 10,247 recordings
        # ningbo (Ningbo First Hospital - 12-lead ECG Database), 34,905 recordings

        self.num_recordings_per_dataset = {'CPSC': 6877,
                                           'CPSC_Extra': 3453,
                                           'StPetersburg': 74,
                                           'PTB': 516,
                                           'PTB_XL': 21837,
                                           'Georgia': 10344,
                                           'Chapman_Shaoxing': 10247,
                                           'Ningbo': 34905
                                           }
        self.starting_index_per_dataset = {'CPSC': 0,
                                           'CPSC_Extra': 6877,
                                           'StPetersburg': 10330,
                                           'PTB': 10404,
                                           'PTB_XL': 10920,
                                           'Georgia': 32757,
                                           'Chapman_Shaoxing': 43101,
                                           'Ningbo': 53348
                                           }
        self.ending_index_per_dataset = {'CPSC': 6876,
                                         'CPSC_Extra': 10329,
                                         'StPetersburg': 10403,
                                         'PTB': 10919,
                                         'PTB_XL': 32756,
                                         'Georgia': 43100,
                                         'Chapman_Shaoxing': 53347,
                                         'Ningbo': 88252
                                         }
        self.categories_mapping_dict = {}
        # self.categories_mapping_dict_scored = {}
        for dataset in self.datasets:
            self.categories_mapping_dict[dataset] = {}
            for category in self.categories_full_name:
                self.categories_mapping_dict[dataset][category] = \
                    self.categories_mapping[dataset][self.categories_mapping['Dx'] == category].values[0]
            # self.categories_mapping_dict_scored[dataset] = {}
            # for category in self.categories_full_name_scored:
            #     self.categories_mapping_dict_scored[dataset][category] = \
            #         self.categories_mapping_scored[dataset][self.categories_mapping_scored['Dx'] == category].values[0]
        self.categories_lookup_dict = {
            self.categories_full_name[i]: self.categories_abbreviation[i]
            for i in range(len(self.categories_full_name))
        }
        self.categories_lookup_dict_reverse = {
            self.categories_abbreviation[i]: self.categories_full_name[i]
            for i in range(len(self.categories_full_name))
        }
        # self.categories_lookup_dict_scored = {
        #     self.categories_full_name_scored[i]: self.categories_abbreviation_scored[i]
        #     for i in range(len(self.categories_full_name_scored))
        # }
        # self.categories_lookup_dict_reverse_scored = {
        #     self.categories_abbreviation_scored[i]: self.categories_full_name_scored[i]
        #     for i in range(len(self.categories_full_name_scored))
        # }

    def get_num_of_images_in_category(self, dataset, category):
        return self.categories_mapping_dict[dataset][category]

    def get_full_name_from_abbreviation(self, abbreviation):
        return self.categories_lookup_dict_reverse[abbreviation]

    def get_abbreviation_from_full_name(self, full_name):
        return self.categories_lookup_dict[full_name]

    def get_dataset_name_from_index(self, index):
        for dataset in self.datasets:
            if self.starting_index_per_dataset[dataset] <= index <= self.ending_index_per_dataset[dataset]:
                return dataset

    def get_indexes_for_dataset(self, dataset):
        return self.starting_index_per_dataset[dataset], self.ending_index_per_dataset[dataset]

    def get_num_recordings_for_dataset(self, dataset):
        return self.num_recordings_per_dataset[dataset]

    # def get_num_of_images_in_category_scored(self, dataset, category):
    #     return self.categories_mapping_dict_scored[dataset][category]
    #
    # def get_full_name_from_abbreviation_scored(self, abbreviation):
    #     return self.categories_lookup_dict_reverse_scored[abbreviation]
    #
    # def get_abbreviation_from_full_name_scored(self, full_name):
    #     return self.categories_lookup_dict_scored[full_name]


class Physionet_Dataset(Source_Dataset):
    # This class is used to load a sub-dataset from the Physionet dataset
    # The sub-dataset is defined by the dataset name
    # The dataset name can be one of the following:
    # 'CPSC', 'CPSC_Extra', 'StPetersburg', 'PTB', 'PTB_XL', 'Georgia', 'Chapman_Shaoxing', 'Ningbo'
    # The categories used are the scored categories that are shared between all sub-datasets.
    # The categories are defined by the categories_mapping_scored dictionary
    # https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv
    def __init__(self, dataset_name, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.categories_mapping = CategoriesMapping()
        self.classification_path = config['Physionet_2021_dataset']['Classification_Path']
        self.database_path = config['Physionet_2021_dataset']['Images_Path']
        # check if ecg_format is str or int:
        if isinstance(ecg_format, str):
            self.database_path = self.database_path + '_format_' + ecg_format
        else:
            self.database_path = self.database_path + '_format_' + str(ecg_format)
        self.files_in_database = os.listdir(self.database_path)
        self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]  # remove non png files
        self.files_in_database = sorted(self.files_in_database, key=lambda x: int(x[4:-4]))
        self.starting_index, self.ending_index = self.categories_mapping.get_indexes_for_dataset(self.dataset_name)
        self.files_in_database = [i for i in self.files_in_database if
                                  int(i[4:-4]) in range(self.starting_index, self.ending_index + 1)]
        # self.categories_lookup_dict_scored = self.categories_mapping.categories_lookup_dict_scored
        self.categories_lookup_dict_full = self.categories_mapping.categories_lookup_dict

        self.categories_lookup_dict = categories.dataset_lookup_dicts[self.dataset_name].get_categories_lookup_dict()

        if classification_category in self.categories_lookup_dict:
            classification_category_abbr = self.categories_lookup_dict[classification_category]
            new_classification_category = []
            if isinstance(classification_category_abbr, list):
                new_classification_category.extend(
                    self.categories_mapping.get_full_name_from_abbreviation(category)
                    for category in classification_category_abbr
                )
            else:
                new_classification_category.extend(
                    (
                        self.categories_mapping.get_full_name_from_abbreviation(
                            classification_category_abbr
                        ),
                    )
                )
            if classification_category is not None:
                self.classification_category = new_classification_category

            total_num_images_per_classification_category = {
                category: self.categories_mapping.get_num_of_images_in_category(
                    dataset_name, category
                )
                for category in self.classification_category
            }
            if sum(total_num_images_per_classification_category.values()) > 0:
                self.database_valid_for_category = True
        else:
            self.database_valid_for_category = False
            self.files_in_database = []
            self.classification_category = classification_category
        self.classifications = self.upload_classifications()

    def __len__(self):
        return len(self.files_in_database) if self.database_valid_for_category else 0

    def __getitem__(self, index):
        item = self.files_in_database[index]
        classification_ = self.classifications[item[:-4]]
        if self.stat_only:
            return 0, classification_
        img = cv2.imread(os.path.join(self.database_path, item))
        img = self.Scale_image(img)
        if self.to_equalize_hist:
            img = self.Normalize_image(img)
        return img, classification_

    def upload_classifications(self):
        classifications = {}
        Trues = 0
        Falses = 0
        if self.database_valid_for_category:
            classifications_file = pd.read_csv(self.classification_path)
            for index, row in classifications_file.iterrows():
                filename = row['filename'] + '.png'
                if int(filename[4:-4]) in range(self.starting_index, self.ending_index + 1):
                    for category in self.classification_category:
                        category_abbreviation = None
                        if self.categories_mapping.get_abbreviation_from_full_name(category) is not None:
                            category_abbreviation = self.categories_mapping.get_abbreviation_from_full_name(category)
                        elif self.categories_mapping.get_abbreviation_from_full_name(category) is not None:
                            category_abbreviation = self.categories_mapping.get_abbreviation_from_full_name(category)
                        if category_abbreviation is not None and row[category_abbreviation] == 1:
                            classifications[filename[:-4]] = True
                            Trues += 1
                            break
                    if classifications.get(filename[:-4], None) is None:
                        classifications[filename[:-4]] = False
                        Falses += 1
        print(
            f'Database Name: {self.dataset_name}, Classification Category: {self.classification_category}, Trues: {Trues}, Falses: {Falses}')
        self.stats = [Trues, Falses]
        return classifications


class CPSC_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='CPSC', classification_category=classification_category, ecg_format=ecg_format,
                         **kwargs)


class CPSC_Extra_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='CPSC_Extra', classification_category=classification_category,
                         ecg_format=ecg_format,
                         **kwargs)


class StPetersburg_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='StPetersburg', classification_category=classification_category,
                         ecg_format=ecg_format,
                         **kwargs)


class PTB_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='PTB', classification_category=classification_category, ecg_format=ecg_format,
                         **kwargs)


class PTB_XL_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='PTB_XL', classification_category=classification_category, ecg_format=ecg_format,
                         **kwargs)


class Georgia_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='Georgia', classification_category=classification_category, ecg_format=ecg_format,
                         **kwargs)


class Chapman_Shaoxing_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='Chapman_Shaoxing', classification_category=classification_category,
                         ecg_format=ecg_format,
                         **kwargs)


class Ningbo_Dataset(Physionet_Dataset):
    def __init__(self, classification_category=None, ecg_format=0, **kwargs):
        super().__init__(dataset_name='Ningbo', classification_category=classification_category, ecg_format=ecg_format,
                         **kwargs)


class SPH_Dataset(Source_Dataset):
    """
    1. contains 25770 ECG records from 24666 patients (55.36% male and 44.64% female), with between 10 and 60 seconds
    2. sampling frequency is 500 Hz
    3. records were acquired from Shandong Provincial Hospital (SPH) between 2019/08 and 2020/08
    4. diagnostic statements of all ECG records are in full compliance with the AHA/ACC/HRS recommendations, consisting of 44 primary statements and 15 modifiers
    5. 46.04% records in the dataset contain ECG abnormalities, and 14.45% records have multiple diagnostic statements
    6. (IMPORTANT) noises caused by the power line interference, baseline wander, and muscle contraction have been removed by the machine
    7. (Label production) The ECG analysis system automatically calculate nine ECG features for reference, which include heart rate, P wave duration, P-R interval, QRS duration, QT interval, corrected QT (QTc) interval, QRS axis, the amplitude of the R wave in lead V5 (RV5), and the amplitude of the S wave in lead V1 (SV1). A cardiologist made the final diagnosis in consideration of the patient health record.
    """

    def __init__(self, classification_category=None, **kwargs):
        super().__init__(classification_category=classification_category, **kwargs)
        self.dataset_name = 'SPH'
        self.database_path = config['SPH_dataset']['Images_Path']
        self.classification_path = config['SPH_dataset']['Classification_Path']
        self.files_in_database = os.listdir(self.database_path)
        self.files_in_database = [file for file in self.files_in_database if file.endswith('.png')]
        self.categories_lookup_dict = categories.dataset_lookup_dicts[self.dataset_name].get_categories_lookup_dict()

        self.classification_category = self.categories_lookup_dict.get(
            classification_category
        )
        if self.classification_category is None:
            self.classification_category = classification_category
        self.database_valid_for_category = classification_category in self.categories_lookup_dict.keys()
        self.classifications = self.upload_classifications()

    def __len__(self):
        return len(self.files_in_database) if self.database_valid_for_category else 0

    def __getitem__(self, index):
        if not self.database_valid_for_category:
            raise Exception('The database is not valid for the selected category')
        file_name = self.files_in_database[index]
        classification_ = self.classifications[file_name[:-4]]
        if self.stat_only:
            return 0, classification_
        img = cv2.imread(os.path.join(self.database_path, file_name))
        img = self.Scale_image(img)
        if self.to_equalize_hist:
            img = self.Normalize_image(img)
        return img, classification_

    def upload_classifications(self):
        classifications = {}
        Trues = 0
        Falses = 0
        if self.database_valid_for_category:
            classifications_file = pd.read_csv(self.classification_path)
            for index, row in classifications_file.iterrows():
                filename = row['filename'] + '.png'
                if filename in self.files_in_database:
                    for category in self.classification_category:
                        if row[category] == 1:
                            classifications[filename[:-4]] = True
                            Trues += 1
                            break
                        if classifications.get(filename[:-4]) is None:
                            classifications[filename[:-4]] = False
                            Falses += 1
        print(
            f'Database name: {self.dataset_name}, category: {self.classification_category}, Trues: {Trues}, Falses: {Falses}')
        self.stats = [Trues, Falses]
        return classifications


class DatasetStatistics:
    def __init__(self, datasets):
        self.datasets = datasets
        self._dataset_statistics()

    def _dataset_statistics(self):
        self.stats = {dataset.dataset_name: dataset.stats for dataset in self.datasets}

    def get_stats(self):
        return self.stats

    def print_stats(self):
        for dataset in self.datasets:
            print(f'Dataset name: {dataset.dataset_name}, Trues: {dataset.stats[0]}, Falses: {dataset.stats[1]}')


def get_datasets():
    datasets = [CPSC_Dataset, CPSC_Extra_Dataset, PTB_Dataset, PTB_XL_Dataset, Georgia_Dataset,
                Chapman_Shaoxing_Dataset, Ningbo_Dataset, SPH_Dataset, Brazilian_Dataset, NY_Dataset]

    categories = ['1st degree AV block', 'AV Block', 'AV Block - Second-degree', 'AV dissociation',
                  'AV junctional rhythm', 'AV node reentrant tachycardia', 'AV reentrant tachycardia',
                  'Abnormal P-wave axis', 'Abnormal QRS', 'Accelerated atrial escape rhythm',
                  'Accelerated idioventricular rhythm', 'Accelerated junctional rhythm', 'Acute myocardial infarction',
                  'Acute myocardial ischemia', 'Acute pericarditis', 'Anterior ischemia',
                  'Anterior myocardial infarction', 'Atrial bigeminy', 'Atrial escape beat', 'Atrial fibrillation',
                  'Atrial fibrillation and flutter', 'Atrial flutter', 'Atrial hypertrophy', 'Atrial pacing pattern',
                  'Atrial premature complex(es) - APC APB', 'Atrial rhythm', 'Atrial tachycardia',
                  'Blocked premature atrial contraction', 'Brady-tachy syndrome', 'Bradycardia', 'Brugada',
                  'Bundle branch block', 'Cardiac dysrhythmia', 'Chronic atrial fibrillation',
                  'Chronic myocardial ischemia', 'Clockwise or counterclockwise vectorcardiographic loop',
                  'Clockwise rotation', 'Complete Left bundle branch block', 'Complete Right bundle branch block',
                  'Complete heart block', 'Coronary heart disease', 'Countercolockwise rotation',
                  'Decreased QT interval', 'Diffuse intraventricular block', 'Early repolarization',
                  'Ectopic atrial rhythm', 'Ectopic atrial tachycardia', 'Electrode reversal', 'FQRS wave',
                  'Fusion beats', 'Fusion beats', 'Heart failure', 'Heart valve disorder', 'High T-voltage',
                  'Idioventricular rhythm', 'Incomplete Left bundle branch block',
                  'Incomplete right bundle branch block', 'Indeterminate cardiac axis',
                  'Inferior ST segment depression', 'Inferior ischaemia', 'Ischemia', 'Junctional escape',
                  'Junctional premature complex', 'Junctional tachycardia', 'Lateral ischemia',
                  'Left anterior fascicular block', 'Left atrial abnormality', 'Left atrial enlargement',
                  'Left atrial hypertrophy', 'Left axis deviation', 'Left posterior fascicular block',
                  'Left ventricular high voltage', 'Left ventricular hypertrophy', 'Left ventricular strain',
                  'Low QRS voltages', 'Mobitz type I wenckebach atrioventricular block',
                  'Mobitz type II atrioventricular block', 'Myocardial infarction', 'Myocardial ischemia',
                  'Nonspecific intraventricular conduction disorder', 'Normal sinus rhythm', 'Normal variant',
                  'Old myocardial infarction', 'P wave changes', 'PR Interval - Prolonged', 'PR Interval - Short',
                  'Pacing', 'Paired ventricular premature complexes', 'Paroxysmal atrial fibrillation',
                  'Paroxysmal supraventricular tachycardia', 'Paroxysmal ventricular tachycardia',
                  'Poor R wave progression', 'Premature ventricular contractions', 'Prolonged P wave',
                  'Prolonged PR interval', 'Prolonged QT interval', 'Pulmonary disease', 'Q wave abnormal',
                  'QT Interval - Prolonged', 'R wave abnormal', 'Rapid atrial fibrillation', 'Right atrial abnormality',
                  'Right atrial enlargement', 'Right atrial high voltage', 'Right atrial hypertrophy',
                  'Right axis deviation', 'Right superior axis', 'Right ventricular hypertrophy', 'ST changes',
                  'ST changes - Nonspecific ST deviation', 'ST changes - Nonspecific ST deviation with T-wave change',
                  'ST changes - Nonspecific T-wave abnormality', 'ST elevation', 'ST interval abnormal', 'STEMI',
                  'STEMI - Anterior', 'STEMI - Anteroseptal', 'STEMI - Inferior or Inferolateral', 'STEMI - Lateral',
                  'STEMI - Posterior', 'STEMI - Right Ventricular', 'Sinosatrial block', 'Sinus arrhythmia',
                  'Sinus atrium to atrial wandering rhythm', 'Sinus bradycardia', 'Sinus node dysfunction',
                  'Sinus tachycardia', 'Supraventricular bigeminy', 'Supraventricular tachycardia',
                  'Suspect arm ecg leads reversed', 'T wave abnormal', 'T wave inversion', 'TU fusion', 'Tachycardia',
                  'Tall P wave', 'Transient ischemic attack', 'U wave abnormal', 'Ventricular bigeminy',
                  'Ventricular ectopics', 'Ventricular escape beat', 'Ventricular escape rhythm',
                  'Ventricular fibrillation', 'Ventricular flutter', 'Ventricular hypertrophy',
                  'Ventricular pacing pattern', 'Ventricular pre excitation', 'Ventricular tachycardia',
                  'Ventricular trigeminy', 'Wandering atrial pacemaker', 'Wolff-Parkinson-White']

    dataset_by_category = {category: [] for category in categories}
    for dataset, category in itertools.product(datasets, categories):
        dataset_by_category[category].append(dataset(category))
    return dataset_by_category


def get_datasets_statistics():
    dataset_by_category = get_datasets()
    return {
        category: DatasetStatistics(dataset_by_category[category])
        for category in dataset_by_category
    }


def plot_statistics():
    datasets_statistics = get_datasets_statistics()
    stats = {category: dataset_stats.get_stats() for category, dataset_stats in datasets_statistics.items()}
    data = pd.DataFrame(stats).T


def save_heatmap():
    datasets_statistics = get_datasets_statistics()
    stats = {category: dataset_stats.get_stats() for category, dataset_stats in datasets_statistics.items()}
    data = pd.DataFrame(stats).T
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(data, annot=True, fmt='g', cmap='Blues', linewidths=.5, ax=ax)
    ax.set_title('Datasets statistics')
    fig.savefig('datasets_statistics.png')
    data_true = data.applymap(lambda x: x[0])
    data_true['Total'] = data_true.sum(axis=1)
    data_true = data_true.drop('Normal sinus rhythm')
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.set_title('Datasets statistics')
    sns.heatmap(data_true, cmap='rocket_r'
                , annot=True, fmt='d', linewidths=.3, cbar=True, ax=ax)
    for t in ax.texts:
        t.set_text(t.get_text() if int(t.get_text()) > 0 else "")

    plt.savefig('Heatmap.png', dpi=300, bbox_inches='tight')


class Combined_Dataset(Source_Dataset):
    # Combining all source datasets. Currently it's NY + Brazilian + Physionet + SPH
    def __init__(self, ecg_format=0, for_adversarial=False, which_dbs_to_use = [True,True,True,True,True,True,True,True,True,True], **kwargs):
        super().__init__(**kwargs)
        physionet_subsets = [CPSC_Dataset, CPSC_Extra_Dataset, PTB_Dataset, PTB_XL_Dataset, Georgia_Dataset,
                             Chapman_Shaoxing_Dataset, Ningbo_Dataset]
        physionet_datasets = [*physionet_subsets]
        physionet_datasets_list = [dataset(ecg_format=ecg_format, **kwargs) for dataset, use in zip(physionet_datasets,which_dbs_to_use[3:]) if use]
        self.datasets_list = []
        if which_dbs_to_use[0]:
            self.datasets_list.append(NY_Dataset(**kwargs))
        if which_dbs_to_use[1]:
            self.datasets_list.append(Brazilian_Dataset(**kwargs))
        if which_dbs_to_use[2]:
            self.datasets_list.append(SPH_Dataset(**kwargs))
        self.datasets_list = [*self.datasets_list,
                              *physionet_datasets_list]
        self.calc_stat()
        self.valid_set = [True if len(x) else False for x in self.datasets_list]
        print(
            f'In total, category: {self.classification_category}, Total length: {self.__len__()}, Trues: {self.stats[0]}, Falses: {self.stats[1]}')

        if not for_adversarial:
            lengths, trues, falses = {}, {}, {}
            if which_dbs_to_use[0]:
                lengths['NY_Dataset'] = len(self.datasets_list[0])
                trues['NY_Dataset'] = self.datasets_list[0].stats[0]
                falses['NY_Dataset'] = self.datasets_list[0].stats[1]
            if which_dbs_to_use[1]:
                lengths['Brazilian_Dataset'] = len(self.datasets_list[1])
                trues['Brazilian_Dataset'] = self.datasets_list[1].stats[0]
                falses['Brazilian_Dataset'] = self.datasets_list[1].stats[1]
            if which_dbs_to_use[2]:
                lengths['SPH_Dataset'] = len(self.datasets_list[2])
                trues['SPH_Dataset'] = self.datasets_list[2].stats[0]
                falses['SPH_Dataset'] = self.datasets_list[2].stats[1]
            for dataset, to_use in zip(physionet_datasets, which_dbs_to_use[3:]):
                if to_use:
                    lengths[dataset.__name__] = len(self.datasets_list[physionet_datasets.index(dataset) + 3])
                    trues[dataset.__name__] = self.datasets_list[physionet_datasets.index(dataset) + 3].stats[0]
                    falses[dataset.__name__] = self.datasets_list[physionet_datasets.index(dataset) + 3].stats[1]

            lengths['Total'] = sum(lengths.values())
            trues['Total'] = sum(trues.values())
            falses['Total'] = sum(falses.values())
            stats_dictionary = {'Datasets Lengths': lengths, 'True Label Counts': trues, 'False Label Counts': falses,
                                'ECG Format': ecg_format}
            if wandb.run is not None:
                wandb.config.update({
                                        'Combined Dataset (scanned images - for training disease classifier) Statistics': stats_dictionary})
                wandb.config.update({
                    'Datasets Used [NY, Brazilian, SPH, CPSC, CPSC_Extra, PTB, PTB_XL, Georgia, Chapman_Shaoxing, Ningbo]': which_dbs_to_use
                })

    def __len__(self):
        l = 0
        for item in self.datasets_list:
            l += len(item)
        return l

    def __getitem__(self, index):
        offset = 0
        for ds in self.datasets_list:
            if index - offset < len(ds):
                return ds[index - offset]
            else:
                offset += len(ds)

    def set_statistics_only(self, stat_only=False):
        for item in self.datasets_list:
            item.stat_only = stat_only


class Format_Physionet_Dataset(Source_Dataset):
    # A combined physionet dataset of ecg_format as defined
    def __init__(self, ecg_format=0, start_index=None, end_index=None, **kwargs):
        super().__init__(**kwargs)

        physionet_subsets = [CPSC_Dataset, CPSC_Extra_Dataset, PTB_Dataset, PTB_XL_Dataset, Georgia_Dataset,
                             Chapman_Shaoxing_Dataset, Ningbo_Dataset]
        datasets = [*physionet_subsets]
        self.datasets_list = [dataset(ecg_format=ecg_format, **kwargs) for dataset in datasets]
        self.start_index = start_index
        self.end_index = end_index
        self.calc_stat()
        # self.valid_set = [True if len(x) else False for x in self.datasets_list]

        print(
            f'In total, category: {self.classification_category}, Total length: {self.__len__()}, Trues: {self.stats[0]}, Falses: {self.stats[1]}')

    def __len__(self):
        l = 0
        for item in self.datasets_list:
            l += len(item)
        if self.start_index is not None:
            l = min(l, self.end_index - self.start_index)
        return l

    def __getitem__(self, index):
        index = index + self.start_index if self.start_index is not None else index
        offset = 0
        for ds in self.datasets_list:
            if index - offset < len(ds):
                return ds[index - offset]
            else:
                offset += len(ds)

    def set_statistics_only(self, stat_only=False):
        for item in self.datasets_list:
            item.stat_only = stat_only


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
    def __init__(self, classification_category=None, which_dbs_to_use = [True,True,True,True,True,True,True,True,True,True],**kwargs):
        super().__init__(**kwargs)
        self.which_dbs_to_use = which_dbs_to_use
        self.source_dataset = Combined_Dataset(classification_category=classification_category, ecg_format=0,
                                               for_adversarial=True,which_dbs_to_use=which_dbs_to_use, **kwargs)
        self.mobile_dataset = Mobile_Dataset()
        print(f'Adversarial database length: {self.__len__()}')
        lengths = {'Source Dataset (scanned images) Length': len(self.source_dataset),
                   'Mobile Dataset (mobile captured images) Length': len(self.mobile_dataset),
                   'Adversarial Length (2*min(mobile,source))': self.__len__()}
        wandb.config.update({'Adversarial Dataset Statistics': lengths})

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
    if tag == 'Combined_Dataset':
        ds = Dataset(classification_category='Bundle branch block')
    else:
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
    print(f'All items are accessible statistically')
    ds.set_statistics_only(False)
    item_0, classification_ = ds[0]
    print(f'Item 0, shape : {np.shape(item_0)}, classification: {classification_}')
    item_1, classification_ = ds[1]
    print(f'Item 1, shape : {np.shape(item_1)}, classification: {classification_}')
    assert np.shape(item_0) == np.shape(item_1), 'Dimensions of subsequent frames are not the same'
    item_, classification_ = ds[len(ds) - 1]
    print(f'Item {len(ds) - 1}, shape : {np.shape(item_)}, classification: {classification_}')
    print(f'Total length: {len(ds)}, Trues: {ds.stats[0]}, Falses: {ds.stats[1]}')
    assert len(ds) == ds.stats[0] + ds.stats[1], 'Total length is not equal to sum of trues and falses'


def Test_database_constructor():
    ds = Combined_Dataset(classification_category='Sinus tachycardia', to_cut_image=True, to_equalize_hist=False,
                          to_use_transform=False)
    print(f'Finished testing of database constructor')


def Test_Datasets():
    print(f'Testing datasets...')
    Test_Specific_Dataset(Combined_Dataset, 'Combined_Dataset')
    Test_Specific_Dataset(NY_Dataset, 'NY_Dataset')
    Test_Specific_Dataset(Brazilian_Dataset, 'Brazilian_Dataset')
    Test_Specific_Dataset(Physionet_Dataset, 'Physionet_Dataset')
    Test_Specific_Dataset(Mobile_Dataset, 'Mobile_Dataset')
    Test_Specific_Dataset(Adversarial_Dataset, 'Adversarial_Dataset')
    print(f'Finished testing datasets...')


def Statistics_all_datasets_all_diseases():
    diseases_list = \
        [
            '1st degree AV block',
            'AV Block',
            'AV Block - Second-degree',
            'AV dissociation',
            'AV junctional rhythm',
            'AV node reentrant tachycardia',
            'AV reentrant tachycardia',
            'Abnormal P-wave axis',
            'Abnormal QRS',
            'Accelerated atrial escape rhythm',
            'Accelerated idioventricular rhythm',
            'Accelerated junctional rhythm',
            'Acute myocardial infarction',
            'Acute myocardial ischemia',
            'Acute pericarditis',
            'Anterior ischemia',
            'Anterior myocardial infarction',
            'Atrial bigeminy',
            'Atrial escape beat',
            'Atrial fibrillation',
            'Atrial fibrillation and flutter',
            'Atrial flutter',
            'Atrial hypertrophy',
            'Atrial pacing pattern',
            'Atrial premature complex(es) - APC APB',
            'Atrial rhythm',
            'Atrial tachycardia',
            'Blocked premature atrial contraction',
            'Brady-tachy syndrome',
            'Bradycardia',
            'Brugada',
            'Bundle branch block',
            'Cardiac dysrhythmia',
            'Chronic atrial fibrillation',
            'Chronic myocardial ischemia',
            'Clockwise or counterclockwise vectorcardiographic loop',
            'Clockwise rotation',
            'Complete Left bundle branch block',
            'Complete Right bundle branch block',
            'Complete heart block',
            'Coronary heart disease',
            'Countercolockwise rotation',
            'Decreased QT interval',
            'Diffuse intraventricular block',
            'Early repolarization',
            'Ectopic atrial rhythm',
            'Ectopic atrial tachycardia',
            'Electrode reversal',
            'FQRS wave',
            'Fusion beats',
            'Fusion beats',
            'Heart failure',
            'Heart valve disorder',
            'High T-voltage',
            'Idioventricular rhythm',
            'Incomplete Left bundle branch block',
            'Incomplete right bundle branch block',
            'Indeterminate cardiac axis',
            'Inferior ST segment depression',
            'Inferior ischaemia',
            'Ischemia',
            'Junctional escape',
            'Junctional premature complex',
            'Junctional tachycardia',
            'Lateral ischemia',
            'Left anterior fascicular block',
            'Left atrial abnormality',
            'Left atrial enlargement',
            'Left atrial hypertrophy',
            'Left axis deviation',
            'Left posterior fascicular block',
            'Left ventricular high voltage',
            'Left ventricular hypertrophy',
            'Left ventricular strain',
            'Low QRS voltages',
            'Mobitz type I wenckebach atrioventricular block',
            'Mobitz type II atrioventricular block',
            'Myocardial infarction',
            'Myocardial ischemia',
            'Nonspecific intraventricular conduction disorder',
            'Normal sinus rhythm',
            'Normal variant',
            'Old myocardial infarction',
            'P wave changes',
            'PR Interval - Prolonged',
            'PR Interval - Short',
            'Pacing',
            'Paired ventricular premature complexes',
            'Paroxysmal atrial fibrillation',
            'Paroxysmal supraventricular tachycardia',
            'Paroxysmal ventricular tachycardia',
            'Poor R wave progression',
            'Premature ventricular contractions',
            'Prolonged P wave',
            'Prolonged PR interval',
            'Prolonged QT interval',
            'Pulmonary disease',
            'Q wave abnormal',
            'QT Interval - Prolonged',
            'R wave abnormal',
            'Rapid atrial fibrillation',
            'Right atrial abnormality',
            'Right atrial enlargement',
            'Right atrial high voltage',
            'Right atrial hypertrophy',
            'Right axis deviation',
            'Right superior axis',
            'Right ventricular hypertrophy',
            'ST changes',
            'ST changes - Nonspecific ST deviation',
            'ST changes - Nonspecific ST deviation with T-wave change',
            'ST changes - Nonspecific T-wave abnormality',
            'ST elevation',
            'ST interval abnormal',
            'STEMI',
            'STEMI - Anterior',
            'STEMI - Anteroseptal',
            'STEMI - Inferior or Inferolateral',
            'STEMI - Lateral',
            'STEMI - Posterior',
            'STEMI - Right Ventricular',
            'Sinosatrial block',
            'Sinus arrhythmia',
            'Sinus atrium to atrial wandering rhythm',
            'Sinus bradycardia',
            'Sinus node dysfunction',
            'Sinus tachycardia',
            'Supraventricular bigeminy',
            'Supraventricular tachycardia',
            'Suspect arm ecg leads reversed',
            'T wave abnormal',
            'T wave inversion',
            'TU fusion',
            'Tachycardia',
            'Tall P wave',
            'Transient ischemic attack',
            'U wave abnormal',
            'Ventricular bigeminy',
            'Ventricular ectopics',
            'Ventricular escape beat',
            'Ventricular escape rhythm',
            'Ventricular fibrillation',
            'Ventricular flutter',
            'Ventricular hypertrophy',
            'Ventricular pacing pattern',
            'Ventricular pre excitation',
            'Ventricular tachycardia',
            'Ventricular trigeminy',
            'Wandering atrial pacemaker',
            'Wolff-Parkinson-White'
        ]

    with open("All databases statistics.txt", "a") as myfile:
        myfile.write("Disease\tPositives\tNegatives\tTotal\tNum_of_databases\n")
    for disease in diseases_list:
        print(f'Testing: {disease}')
        ds = Combined_Dataset(classification_category=disease, to_cut_image=True, to_equalize_hist=False,
                              to_use_transform=False)
        with open("All databases statistics.txt", "a") as myfile:
            myfile.write(
                f"{disease}\t{ds.stats[0]}\t{ds.stats[1]}\t{ds.stats[0] + ds.stats[1]}\t{np.sum(ds.valid_set)}\n")


if __name__ == "__main__":
    Statistics_all_datasets_all_diseases()
    # Test_database_constructor()
    # Test_Datasets()
