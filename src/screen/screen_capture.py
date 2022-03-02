import subprocess
import re
from typing import Union, List

import cv2 as cv
import numpy as np
from PIL import Image
from Xlib import display, X


class ScreenCapture:
    """
    Handles functionality related to screen capture for drawings
    """

    def __init__(
        self,
        unique_window_keywords: Union[str, List[str]] = ["skribbl", "brave"],
    ):
        """
        Initialize ScreenCapture class
        """
        self.unique_window_keywords = unique_window_keywords
        self.dsp = display.Display()

        self.window_id = self._get_window_id()
        assert (
            len(self.window_id) == 10
        ), "Check unique_window_keywords and make sure the window is visible on screen"

        self.window_pos = self._get_window_pos()
        self.drawing_area_bbox = self._find_bbox_of_drawing_area()

        x, y, w, h = self.drawing_area_bbox
        print(
            "Drawing area captured is shown in new window, Continue by pressing any key"
            " or Cancel using Ctrl+C and Adjust screen to make drawing area larger"
        )
        cv.imshow("Drawing Area Captured", self._screenshot()[y : y + h, x : x + w])
        cv.waitKey(0)
        cv.destroyWindow("Drawing Area Captured")

    def capture_drawing_area(self):
        """
        Take screenshot of the drawing area every 1 second
        """
        shot = self._screenshot()
        x, y, w, h = self.drawing_area_bbox
        return shot[y : y + h, x : x + w]

    def _screenshot(self) -> np.ndarray:
        """
        Take a screenshot of the entire window
        """
        win_x, win_y, width, height = (
            self.window_pos[0],
            self.window_pos[1],
            self.window_pos[2],
            self.window_pos[3],
        )
        try:
            root = self.dsp.screen().root
            raw = root.get_image(win_x, win_y, width, height, X.ZPixmap, 0xFFFFFFFF)
            image = Image.frombytes("RGB", (width, height), raw.data, "raw", "BGRX")
            image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
            return image
        except Exception as e:
            print(e)
            print("Some error occured when taking screenshot")
            return np.ndarray([])

    def _get_window_id(self):
        """
        Gets the window id
        """
        windows_info = (
            subprocess.run(["wmctrl", "-l"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .lower()
        )
        window_info_list = windows_info.split("\n")

        window_info = self._find_window(window_info_list)

        return window_info[:10]

    def _find_window(self, window_info_list: list) -> str:
        """
        Check if each window info contains all unique search keywords
        """
        for window_info in window_info_list:
            is_found = all(item in window_info for item in self.unique_window_keywords)
            if is_found:
                return window_info

        return ""

    def _get_window_pos(self):
        """
        Get the window position
        """
        win_info = (
            subprocess.run(["xwininfo", "-id", self.window_id], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .split("\n")
        )

        return self._find_bbox_of_window(win_info)

    def _find_bbox_of_window(self, window_info: List[str]) -> tuple:
        """
        Find the position of window on screen and return the (x, y, w, h) coordinates
        """
        num_regex = r"\d+"
        abs_x, abs_y, width, height = 0, 0, 0, 0
        for info in window_info:
            if "Absolute" in info and "X" in info:
                temp = re.search(num_regex, info)
                if temp:
                    abs_x = int(temp.group(0))
            if "Absolute" in info and "Y" in info:
                temp = re.search(num_regex, info)
                if temp:
                    abs_y = int(temp.group(0))
            if "Width" in info:
                temp = re.search(num_regex, info)
                if temp:
                    width = int(temp.group(0))
            if "Height" in info:
                temp = re.search(num_regex, info)
                if temp:
                    height = int(temp.group(0))

        return (abs_x, abs_y, width, height)

    def _find_bbox_of_drawing_area(self):
        """
        Find the bbox of drawing area inside the window and use it for the rest of the
        program
        """
        first_screenshot = self._screenshot()

        screenshot_gray = cv.cvtColor(first_screenshot, cv.COLOR_BGR2GRAY)
        ret, screenshot_thresholded = cv.threshold(screenshot_gray, 225, 255, 0)
        contours, hierarchy = cv.findContours(
            screenshot_thresholded, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) > 0, "Contours not found"
        largest_contour = max(contours, key=cv.contourArea)

        x, y, w, h = cv.boundingRect(largest_contour)

        return (x, y, w, h)


# debug
if __name__ == "__main__":
    screen_capture = ScreenCapture()

    while True:
        drawing_area_shot = screen_capture.capture_drawing_area()

        # make a small window to check what is being shown to the model
        drawing_area_resized = cv.resize(
            drawing_area_shot,
            (drawing_area_shot.shape[1] // 3, drawing_area_shot.shape[0] // 3),
        )
        cv.imshow("Drawing", drawing_area_resized)

        if cv.waitKey(1000) == ord("q"):
            break

    cv.destroyAllWindows()
