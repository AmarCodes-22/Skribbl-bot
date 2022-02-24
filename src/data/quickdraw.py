import os
import struct
from struct import unpack

import cairocffi as cairo
from google.cloud import storage
import matplotlib.image as mpimg
import numpy as np


class Quickdraw:
    def __init__(self, labels_fpath: str) -> None:
        self.bucket_name = "quickdraw_dataset"
        self.labels = self._load_labels(labels_fpath)

    def download_binary_format(self, class_names: list, dest_dir: str) -> None:
        """Downloads binary format of the dataset into dest_dir

        Args:
            class_names (list): List of class names to download binary files for
            dest_dir (str): Path to directory where files will be downloaded
        """
        assert dest_dir is not None
        assert len(class_names) > 0

        self._create_dir_if_not_exists(dest_dir)
        # if not os.path.exists(dest_dir):
        #     print(f"Creating directory {dest_dir}")
        #     os.makedirs(dest_dir)

        for class_name in class_names:
            blob_name = f"full/binary/{class_name}.bin"
            dest_fname = os.path.join(dest_dir, f"{class_name}.bin")

            if not os.path.exists(dest_fname):
                self._download_public_file(
                    source_blob_name=blob_name, destination_fname=dest_fname
                )
            else:
                pass
                print(f"\n The file {dest_fname} already exists. Skipping Download")

    def load_binary_as_strokes(self, binary_fpath: str) -> list:
        """Loads a binary file into a list of drawings

        Args:
            binary_fpath (str): path to the binary file to be loaded

        Returns:
            list: List of dictionaries where each dictionary contains a single
            drawing
        """
        assert binary_fpath is not None

        class_name = os.path.basename(binary_fpath)[:-4]
        print(f"Loading images for class {class_name}")
        drawings = list()

        with open(binary_fpath, "rb") as f:
            while True:
                try:
                    drawing = self._unpack_drawing(f)
                    drawing.update({"word": class_name})
                    drawings.append(drawing)
                except struct.error:
                    break
        return drawings

    def convert_strokes_to_image(
        self,
        strokes,
        side=256,
        line_diameter=4,
        bg_color=(1, 1, 1),
        fg_color=(0, 0, 0),
    ) -> np.ndarray:
        """Convert stroke to Numpy arrays

        Args:
            strokes (list): A list of strokes where (x, y) points are provided
            side (int, optional): Length of side of an image. Defaults to 256.
            line_diameter (int, optional): Width of strokes. Defaults to 4.
            bg_color (tuple, optional): Background color for image.
                Defaults to (1, 1, 1).
            fg_color (tuple, optional): Foreground color for stroke.
                Defaults to (0, 0, 0).

        Returns:
            np.ndarray: Array of pixel intensities
        """
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
        ctx = cairo.Context(surface)

        ctx = self._set_options_for_context(ctx)

        ctx.set_line_width(line_diameter)
        ctx.scale(0.9, 0.9)

        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        centered_image = self._center_image(strokes, side)

        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered_image:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        image_data = surface.get_data()
        image = self._convert_raster_to_nparray(image_data, side)

        return image

    def save_image_to_disk(self, image: np.ndarray, fpath: str) -> None:
        """Saved numpy array to file path provided

        Args:
            image (np.ndarray): Numpy array of pixel intensities
            fpath (str): path to the file where image is to be stored
        """
        # print(f"Saving image to {fpath}")
        mpimg.imsave(fpath, image, cmap="gray")

    def _download_public_file(
        self, source_blob_name: str, destination_fname: str, bucket_name: str = ""
    ) -> None:
        """Download file from public file storage

        Args:
            source_blob_name (str): Blob name from the file URL
            destination_fname (str): Path where to download the file
            bucket_name (str, optional): GCS bucket name. Defaults to None.
        """
        bucket_name = self.bucket_name
        storage_client = storage.Client.create_anonymous_client()

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_fname)
        print(
            "Downloading public blob {}\n\t from bucket {}\n\t to {}".format(
                source_blob_name, bucket.name, destination_fname
            )
        )

    def _load_labels(self, labels_file_path: str) -> list:
        labels_file = open(labels_file_path, "r")
        return labels_file.read().splitlines()

    def _unpack_drawing(self, file_handle) -> dict:
        (key_id,) = unpack("Q", file_handle.read(8))
        (country_code,) = unpack("2s", file_handle.read(2))
        (recognized,) = unpack("b", file_handle.read(1))
        (timestamp,) = unpack("I", file_handle.read(4))
        (n_strokes,) = unpack("H", file_handle.read(2))

        strokes = []
        for i in range(n_strokes):
            (n_points,) = unpack("H", file_handle.read(2))
            fmt = str(n_points) + "B"
            x = unpack(fmt, file_handle.read(n_points))
            y = unpack(fmt, file_handle.read(n_points))
            strokes.append((x, y))

        return {
            "country_code": country_code,
            "recognized": recognized,
            "timestamp": timestamp,
            "strokes": strokes,
        }

    def _set_options_for_context(self, context) -> cairo.Context:
        context.set_antialias(cairo.ANTIALIAS_BEST)
        context.set_line_cap(cairo.LINE_CAP_ROUND)
        context.set_line_join(cairo.LINE_JOIN_ROUND)
        return context

    def _center_image(self, vector_image, side):
        # center the image
        bbox_tl = np.hstack(vector_image).min(axis=1)
        bbox_br = np.hstack(vector_image).max(axis=1)
        bbox = bbox_br - bbox_tl

        offset = ((side, side) - bbox) // 2
        offset = offset.reshape(-1, 1)
        # ceil(0.05 * 256) = 13 to account for scaling
        centered = [stroke + offset + [[13], [13]] for stroke in vector_image]
        return centered

    def _convert_raster_to_nparray(self, raster, side_len):
        raster_image = np.copy(np.asarray(raster)[::4])
        return raster_image.reshape((side_len, side_len))

    def _create_dir_if_not_exists(self, dir_path: str):
        if not os.path.exists(dir_path):
            print(f"Creating directory {dir_path}")
            os.makedirs(dir_path)
