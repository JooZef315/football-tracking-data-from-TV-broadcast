import sys
sys.path.append("..")


import numpy as np
import tables

from world_cup_data_augmentation import raw_data_loader
from utils import utils

class WorldCupH5Builder():
    def __init__(self, data_dir, dataset_type):
        self.total_samples = 50000
        self.loader = raw_data_loader.RawDataloader(data_dir)
        self.file_name = 'world_cup_{0}.h5'.format(dataset_type)
        self.samples_per_image = 1
        if dataset_type == 'test':
            self.id_range = range(1, 186+1)
        else:
            raise NotImplementedError()

    def init_h5_file(self):
        h5file = tables.open_file(self.file_name, mode="w", title="worldcup dataset")
        filters = tables.Filters(complevel=5, complib='blosc')
        video_storage = h5file.create_earray(
            h5file.root,
            'frames',
            tables.Atom.from_dtype(np.dtype(np.uint8)),
            shape=(0, 256, 256, 3),
            filters=filters,
            expectedrows=10000000)
        homography_storage = h5file.create_earray(
            h5file.root,
            'homographies',
            tables.Atom.from_dtype(np.dtype(np.float64)),
            shape=(0, 3, 3),
            filters=filters,
            expectedrows=10000000)
        storage = [h5file, video_storage, homography_storage]
        return storage

    def append_data(self, storage):
        h5file, video_storage, homography_storage = storage
        for image_id in self.id_range: 
            for _ in range(self.samples_per_image):
                cropped_frame, cropped_homography = self.loader.get_paired_data_by_id(image_id)
                cropped_frame = cropped_frame * 255.0
                cropped_frame = cropped_frame.astype(np.uint8)
                homography_storage.append(cropped_homography[None])
                video_storage.append(cropped_frame[None])
        h5file.close()

    def build_h5(self):
        storage = self.init_h5_file()
        self.append_data(storage)


def main():
    test_builder = WorldCupH5Builder('test', 'test')
    test_builder.build_h5()


if __name__ == '__main__':
    main()
