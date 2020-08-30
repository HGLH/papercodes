import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        #os.path.join 拼接路径
        #glob.glob模式匹配

        self.files_A = sorted(glob.glob(os.path.join(root,"%sA" %mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"%sB" %mode) + "/*.*"))
        #print(self.files_A)

    def __getitem__(self, index):
        image_A=Image.open(self.files_A[index%len(self.files_A)])

        image_B = Image.open(self.files_B[index % len(self.files_B)])

        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A),len(self.files_B))