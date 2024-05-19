

import random
from torchvision import transforms
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
from functools import partial
import math


random_mirror = True

def scale_to_new_range(m, min, max):

    old_range = (1 - 0)
    new_range = (max - min)
    new_value = (((m - 0) * new_range) / old_range) + min

    return new_value

def tensor_to_pil(tensor):

    return transforms.ToPILImage()(tensor.squeeze_(0))

def pil_to_tensor(pil):

    return transforms.ToTensor()(pil)

def ShearX(img, v):  # [-0.3, 0.3]
    v = scale_to_new_range(v, 0, 0.3)
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    v = scale_to_new_range(v, 0, 0.3)
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = scale_to_new_range(v, 0, 0.45)
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = scale_to_new_range(v, 0, 0.45)
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = scale_to_new_range(v, 0, 10)
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = scale_to_new_range(v, 0, 10)
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    v = scale_to_new_range(v, 0, 30)
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def Rotate_deterministic(img, v):  # [-30, 30]
    v = scale_to_new_range(v, -30, 30)
    assert -30 <= v <= 30
    return img.rotate(v)


def AutoContrast(img, v):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, v):
    return PIL.ImageOps.invert(img)


def Equalize(img, v):
    return PIL.ImageOps.equalize(img)


def Flip(img, v):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    v = scale_to_new_range(v, 0, 256)
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    v = scale_to_new_range(v, 4, 8)
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    v = scale_to_new_range(v, 0, 4)
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    v = scale_to_new_range(v, 0.1, 1.9)
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    v = scale_to_new_range(v, 0.1, 1.9)
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    v = scale_to_new_range(v, 0.1, 1.9)
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    v = scale_to_new_range(v, 0.1, 1.9)
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    v = scale_to_new_range(v, 0, 0.2)
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img
    # print(img.size)
    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # v = scale_to_new_range(v, 0, 20)
    # assert 0 <= v <= 20
    if v < 0:
        return img

    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    import copy
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


augmentation_list = [
    # partial(ShearX, v=0.5),
    # partial(ShearY, v=0.5),
    # partial(TranslateX, v=0.5),
    # partial(TranslateY, v=0.5),
    # partial(Rotate, v=0.5),
    # partial(AutoContrast, v=0.5),
    # partial(Invert, v=0.5),
    # partial(Equalize, v=0.5),
    # partial(Solarize, v=0.5),
    # partial(Posterize, v=0.5),
    # partial(Contrast, v=0.5),
    # partial(Color, v=0.5),
    # partial(Brightness, v=0.5),
    # partial(Sharpness, v=0.5),
    # partial(Cutout, v=0.5),
    # partial(Flip, v=0.5), # instead of sample_pairing, add Flip to remove randomness

    partial(ShearX, v=1),
    partial(ShearY, v=1),
    partial(TranslateX, v=1),
    partial(TranslateY, v=1),
    partial(Rotate, v=1),
    partial(AutoContrast, v=1),
    partial(Invert, v=1),
    partial(Equalize, v=1),
    partial(Solarize, v=1),
    partial(Posterize, v=1),
    partial(Contrast, v=1),
    partial(Color, v=1),
    partial(Brightness, v=1),
    partial(Sharpness, v=1),
    partial(Cutout, v=1),
    partial(Flip, v=1),  # instead of sample_pairing, add Flip to remove randomness
]


class DataAugmentation:

    def __init__(self, aug_str):
        if len(aug_str) != 16 or any(c not in '01' for c in aug_str):
            raise ValueError("Input must be a 15-character string containing only 0s and 1s.")

        self.augs = [c == '1' for c in aug_str]

    def __call__(self, img):

        if not True in self.augs:
            return img

        result_img = np.array(img)
        for i, aug in enumerate(self.augs):
            if aug:
                augmented = augmentation_list[i](img)
                result_img = np.concatenate((result_img, np.array(augmented)), axis=-1)
        return result_img

class DataRotation:
    def __init__(self, target_channel):
        self.target_channel = target_channel
    def __call__(self, img):
        img_array = np.array(img)
        num_channel = img_array.shape[2]  # Assuming img is in HWC format
        num_samples = math.ceil(self.target_channel / num_channel) - 1 # assume the first element is the original image
        mag_list = np.linspace(0, 1, num_samples)
        result_img = np.array(img)
        for i in range(num_samples):
            augmented = Rotate_deterministic(img, v=mag_list[i])
            result_img = np.concatenate((result_img, np.array(augmented)), axis=-1)
        return result_img[:, :, :self.target_channel]

if __name__ == "__main__":
    import torchvision
    from torchvision import transforms, datasets
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming data_reshape is already imported and defined elsewhere

    # Initialize the transformation and the Caltech101 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a common size
    ])

    # Load Caltech101 dataset
    dataset = torchvision.datasets.Caltech101(root='../data/Caltech101', download=True, transform=transform)

    # Load a single image and its label
    img, label = dataset[0]  # Get the first image and label from the dataset

    # Initialize your custom reshape class
    # augmenter = DataAugmentation("000000000000001")  # Example: target to 64x64 image with 9 channels
    augmenter = DataRotation(target_channel=64)

    # Apply the reshaping to the image
    augmented_img = augmenter(img)

    # Plotting the original and augmented images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.array(img))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(np.array(augmented_img[:,:,3:6]))
    axes[1].set_title('Augmented Image')
    axes[1].axis('off')

    plt.show()