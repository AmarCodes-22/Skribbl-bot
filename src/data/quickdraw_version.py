import os
import random
import shutil
from typing import Union
from matplotlib.pyplot import draw

from torch import classes

from src.data.quickdraw import Quickdraw


class QuickdrawVersion(Quickdraw):
    def __init__(
        self,
        labels_fpath: str,
        dir_path: str,
        num_classes: int,
        num_images_per_class: int,
        classes: Union[list, str] = "random",
    ) -> None:
        super().__init__(labels_fpath)

        self.num_images_per_class = num_images_per_class

        if isinstance(classes, list):
            num_classes = len(classes)
            self.classes = classes
        elif isinstance(classes, str):
            assert num_classes is not None
            self.classes = random.sample(self.labels, num_classes)
            print(f"Randomly chosen classes {self.classes}")

        self.dataset_dir = dir_path
        self.binary_files_dir = os.path.join(dir_path, "binary")
        self.images_dir = os.path.join(dir_path, "images")

        self._create_dir_if_not_exists(self.dataset_dir)
        self._create_dir_if_not_exists(self.binary_files_dir)
        self._create_dir_if_not_exists(self.images_dir)

    def create_version_in_directory(self):

        self.download_binary_format(
            class_names=self.classes, dest_dir=self.binary_files_dir
        )

        for binary_fname in os.listdir(self.binary_files_dir):

            class_name = binary_fname[:-4]
            class_images_dir = os.path.join(self.images_dir, class_name)
            self._create_dir_if_not_exists(class_images_dir)

            binary_fpath = os.path.join(self.binary_files_dir, binary_fname)
            drawings = self.load_binary_as_strokes(binary_fpath)

            random_drawings = self._chooose_random_drawings(
                drawings, self.num_images_per_class
            )

            for i, drawing in enumerate(random_drawings):
                strokes = drawing["strokes"]
                img_arr = self.convert_strokes_to_image(strokes)
                self.save_image_to_disk(
                    img_arr, os.path.join(class_images_dir, f"{i}.jpg")
                )

    def split_train_test(self, test_ratio: float = 0.1):
        self.train_dir = os.path.join(self.images_dir, "train")
        self.test_dir = os.path.join(self.images_dir, "test")

        self._create_dir_if_not_exists(self.train_dir)
        self._create_dir_if_not_exists(self.test_dir)

        for class_name in os.listdir(self.images_dir):
            if class_name == "train" or class_name == "test":
                continue

            self._create_dir_if_not_exists(os.path.join(self.train_dir, class_name))
            self._create_dir_if_not_exists(os.path.join(self.test_dir, class_name))

            all_images = os.listdir(os.path.join(self.images_dir, class_name))
            random.shuffle(all_images)

            num_test_images = int(test_ratio * len(all_images))

            test_images = all_images[:num_test_images]
            train_images = all_images[num_test_images:]

            for image in test_images:
                src_fpath = os.path.join(self.images_dir, class_name, image)
                dst_fpath = os.path.join(self.test_dir, class_name, image)
                shutil.move(src_fpath, dst_fpath)

            for image in train_images:
                src_fpath = os.path.join(self.images_dir, class_name, image)
                dst_fpath = os.path.join(self.train_dir, class_name, image)
                shutil.move(src_fpath, dst_fpath)

            os.rmdir(os.path.join(self.images_dir, class_name))

    def _chooose_random_drawings(self, all_drawings: list, num_drawings_to_choose: int):
        return random.sample(all_drawings, num_drawings_to_choose)
