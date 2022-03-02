from io import BytesIO

from PIL import Image

def array_to_bytes(bytes_io, arr):
    raw_image = Image.fromarray(arr)
    raw_image.save(bytes_io, format='PNG')
    bytes_io.seek(0)
    return bytes_io.read()
