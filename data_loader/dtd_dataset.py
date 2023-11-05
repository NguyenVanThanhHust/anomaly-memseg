import os
from os.path import join, isdir, isfile

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DTD_Dataset(Dataset):
    def __init__(self, data_folder, transform = None):
        self.data_folder = join(data_folder, "images")
        categories = next(os.walk(self.data_folder))[1]
        self.im_paths = []
        for cat in categories:
            cat_folder = join(self.data_folder, cat)
            im_files = next(os.walk(cat_folder))[2]
            for im_file in im_files:
                im_path = join(cat_folder, im_file)
                self.im_paths.append(im_path)
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.im_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
            ])
    dtd_dataset = DTD_Dataset(data_folder="../datasets/dtd", transform=base_transform)
    print("num sample: ", dtd_dataset.__len__())