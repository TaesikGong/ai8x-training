import torch
import math
import torchvision


class data_reshape:
    """
    Fold data to increase the number of channels. An interlaced approach used in this folding
    as explained in [1].

    [1] https://arxiv.org/pdf/2203.16528.pdf
    """

    def __init__(self, target_size, target_channel):
        self.target_size = target_size
        self.target_channel = target_channel

    def __call__(self, img):
        current_num_channel = img.shape[0]
        if self.target_channel == current_num_channel and self.target_size == img.shape[1] and self.target_size == \
                img.shape[2]:
            return img

        height_ratio = img.shape[1] / self.target_size
        width_ratio = img.shape[2] / self.target_size

        reshaped_img = torch.zeros((self.target_channel, self.target_size, self.target_size))

        for i in range(self.target_size):
            for j in range(self.target_size):
                start_row = int(i * height_ratio)
                end_row = int(min((i + 1) * height_ratio, img.shape[1]))  # Ensure the index does not go out of bounds
                start_col = int(j * width_ratio)
                end_col = int(min((j + 1) * width_ratio, img.shape[2]))  # Ensure the index does not go out of bounds

                block = img[:, start_row:end_row, start_col:end_col]  # a block

                b_width = end_col - start_col
                b_height = end_row - start_row
                samples_to_take = math.ceil(self.target_channel / 3)
                step_size = b_width * b_height / samples_to_take
                for c in range(samples_to_take):
                    if (c + 1) * 3 > self.target_channel: # for the remaining channels that are not divided by 3
                        gap = (c + 1) * 3 - self.target_channel
                    else:
                        gap = 0

                    reshaped_img[c * 3:(c + 1) * 3, i, j] = block[:3-gap, int(c * step_size) // b_width,
                                                            int(c * step_size) % b_width]  # row-major stepping

        return reshaped_img


if __name__ == "__main__":
    from torchvision import transforms, datasets
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming data_reshape is already imported and defined elsewhere

    # Initialize the transformation and the Caltech101 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a common size
        transforms.ToTensor()  # Convert images to tensor
    ])

    # Load Caltech101 dataset
    dataset = torchvision.datasets.Caltech101(root='../data/Caltech101', download=True, transform=transform)

    # Load a single image and its label
    img, label = dataset[0]  # Get the first image and label from the dataset

    # Initialize your custom reshape class
    reshaper = data_reshape(target_size=32, target_channel=11)  # Example: target to 64x64 image with 9 channels

    # Apply the reshaping to the image
    reshaped_img = reshaper(img)
    print(img.shape)
    print(reshaped_img.shape)


    # Function to display images
    def show_tensor_images(original, transformed):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # Original image
        ax[0].imshow(np.transpose(original.numpy(), (1, 2, 0)))
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Transformed image
        # Since reshaped_img may have more than 3 channels, just show the first 3 for visualization
        ax[1].imshow(np.transpose(transformed.numpy()[6:9], (1, 2, 0)))
        ax[1].set_title('Transformed Image')
        ax[1].axis('off')

        plt.show()


    # Display the images
    show_tensor_images(img, reshaped_img)
