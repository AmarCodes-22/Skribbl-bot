from io import BytesIO
import requests
from time import sleep

from PIL import Image

from src.screen import screen_capture
from src.utils import array_to_bytes


def main():
    window = screen_capture.ScreenCapture()

    while True:
        screenshot = window.capture_drawing_area()

        raw_image_bytes = BytesIO()
        img_bytes = array_to_bytes(raw_image_bytes, screenshot)

        url = "http://localhost:8080/predictions/resnet-18"
        response = requests.post(
            url,
            files={"data": img_bytes},
        )
        if response.status_code == 200:
            output = response.json()
            print(output)

        sleep(1)


if __name__ == "__main__":
    main()
