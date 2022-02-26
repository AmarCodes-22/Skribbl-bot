import os
import random
import shutil

from torchvision import datasets

from src.data.quickdraw import Quickdraw


class QuickdrawVersion(Quickdraw):
    def __init__(self, labels_fpath: str = "", num_images_per_class: int = 0) -> None:
        """Use this class to create new dataset versions or interact with an existing one

        Args:
            labels_fpath (str, optional): Path to text file containing class names to
                generate seperated by newline if creating a new dataset. Optional if
                loading an existing dataset
            num_images_per_class (int, optional): Number of images to generate per
                class, Defaults to 0
        """
        # only when creating a new version
        if len(labels_fpath) > 0:
            super().__init__(labels_fpath)
            if num_images_per_class > 0:
                self.num_images_per_class = num_images_per_class

    def load_from_directory(self, dir_path: str):
        """
        Load a previously created dataset from directory.
        Provide path to the folder containing 'train' and 'test' folders
        """
        self.data_dir = dir_path

        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.data_dir, x),
            )
            for x in ["train", "test"]
        }

        self.labels = image_datasets["train"].classes
        self.num_classes = len(self.labels)
        self.num_images_per_class = (
            len(image_datasets["train"]) // self.num_classes
            + len(image_datasets["test"]) // self.num_classes
        )

        return image_datasets

    def create_version_in_directory(self, dir_path: str):
        """Generates a version in dir_path

        Args:
            dir_path (str): path to the folder where to generate the dataset
        """
        self.data_dir = dir_path
        self._setup_directories(dir_path)

        self.download_binary_format(dest_dir=self.binary_files_dir)

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
        """Splits the images into train and test

        Args:
            test_ratio (float, optional): Ratio of images to use for testing.
                Defaults to 0.1.
        """
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

    def _setup_directories(self, directory_path: str):
        self.dataset_dir = directory_path
        self.binary_files_dir = os.path.join(directory_path, "binary")
        self.images_dir = os.path.join(directory_path, "images")

        self._create_dir_if_not_exists(self.dataset_dir)
        self._create_dir_if_not_exists(self.binary_files_dir)
        self._create_dir_if_not_exists(self.images_dir)
