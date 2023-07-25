import os
from glob import glob
from PIL import Image
import torchvision
import torchvision.transforms.functional as VF
from torchvision.transforms import InterpolationMode
import cv2
import tqdm
from timeit import timeit

torchvision.set_image_backend('accimage')
DATA_ROOT = './files/'

def vision_load_image(path):
    image = Image.open(path).convert('RGB')
    image.load()
    return image

def cv2_load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def vision_crop_resize(image, size):
    image = F.resized_crop(image, 0, 0, 256, 256, (244, 244), InterpolationMode.BILINEAR)
    return image

def cv2_crop_resize(arr, size):
    arr = arr[0:0+256, 0:0+256]
    arr = cv2.resize(arr, (244, 244), cv2.INTER_LINEAR)
    return arr

def vision_hflip(image):
    image = VF.hflip(image)
    return Image

def cv2_hflip(arr):
    arr = cv2.flip(arr, 1)
    return arr

def vision_normalize(arr):
    arr = VF.normalize(arr, (127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    return arr

def cv2_normalize(arr):
    arr = cv2.normalize(arr)
    return arr

def main():
    filepath = glob(os.path.join(DATA_ROOT, '*.JPEG'))
    print(f'{len(filepath)} files found.')

    stmt = '[cv2_load_image(fp) for fp in filepath]'

    timeit(stmt, number=5)

if __name__ == '__main__':
    main()