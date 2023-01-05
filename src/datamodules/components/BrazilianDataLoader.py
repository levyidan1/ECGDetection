from torch.utils.data import Dataset
import h5py


def extract_labels_from_csv(labels_file):
    header = labels_file.readline().split(',')
    header = [x.strip() for x in header]
    labels = []
    for line in labels_file:
        line = line.split(',')
        line[-1] = line[-1].strip()
        labels.append(dict(zip(header, line)))
    return labels


class BaseBrazilianDatabase(Dataset):
    def __init__(self, data_path, labels_path, classification_category='AF'):
        self.data_path = data_path
        self.classification_category = classification_category
        self._in_w = 1650
        self._in_h = 880
        self.__load_labels__(labels_path)

    def __load_data__(self, data_path):
        pass

    def __load_labels__(self, labels_path):
        pass

    def __len__(self):
        pass

    def __get_labels__(self, idx):
        return {'1dAVb': self.labels[idx]['1dAVb'], 'RBBB': self.labels[idx]['RBBB'],
                'LBBB': self.labels[idx]['LBBB'], 'SB': self.labels[idx]['SB'],
                'AF': self.labels[idx]['AF'], 'ST': self.labels[idx]['ST']}

    def __get_classification__(self, idx):
        classification_str = self.labels[idx][self.classification_category]
        return True if classification_str == '1' else False

    def __getitem__(self, idx):
        pass


class BrazilianImageDatabase(BaseBrazilianDatabase):
    def __init__(self, data_path, labels_path, classification_category='AF', number_of_images_to_load=None):
        self.stat_only = False
        self.number_of_image_files = 80  # WIP. Currently only 533,000 images have been created. TODO: Change to 882 when all images are created.
        self.number_of_images_to_load = number_of_images_to_load
        super().__init__(data_path, labels_path, classification_category)

    def __load_data__(self, data_path):
        self.data = {}
        current_image_index = 0
        for i in range(self.number_of_image_files):
            image_data_file = h5py.File(f'{data_path}/images_batch_{i}.hdf5', 'r')
            image_data = image_data_file['images']
            if self.number_of_images_to_load is not None:
                remaining_images = self.number_of_images_to_load - current_image_index
                if remaining_images <= 0:
                    break
                elif remaining_images < len(image_data):
                    image_data = image_data[:remaining_images]
            self.data[i] = image_data
            current_image_index += len(image_data)

    def __load_labels__(self, labels_path):
        num_true_values = 0
        num_false_values = 0
        current_image_index = 0
        for i in range(self.number_of_image_files):
            if self.number_of_images_to_load is not None and current_image_index >= self.number_of_images_to_load:
                break
            with open(f'{labels_path}/labels_batch_{i}.csv', 'r') as labels_file:
                labels = extract_labels_from_csv(labels_file)
                if self.number_of_images_to_load is not None:
                    remaining_labels = self.number_of_images_to_load - current_image_index
                    if remaining_labels <= 0:
                        break
                    elif remaining_labels < len(labels):
                        labels = labels[:remaining_labels]
                if i == 0:
                    self.labels = labels
                else:
                    self.labels.extend(labels)
                for label in labels:
                    if label[self.classification_category] == '1':
                        num_true_values += 1
                    else:
                        num_false_values += 1
                current_image_index += len(labels)
        self.stats = [num_true_values, num_false_values]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not hasattr(self, 'data'):
            self.__load_data__(self.data_path)
        image_file_idx = idx // 1000
        number_of_images = 1000 if image_file_idx < 882 else 944  # a total of 882944 images. (TODO: Verify when all images are created)
        classification_ = self.__get_classification__(idx)
        if self.stat_only:
            return 0, classification_
        img = self.data[image_file_idx][
            idx % number_of_images]  # len(self.data[image_file_idx]) should be 85000. batch 10 (11th) has 32944 images
        return img, classification_

    def set_statistics_only(self, stat_only=False):
        self.stat_only = stat_only
