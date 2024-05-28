import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_loader(rot_dir="D:/2A/Projet_PFA/Drive_data/seg", channels_img=1, image_size=256, batch_size=32):
    # Transofrmations
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=channels_img),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
        ]
    )

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir=rot_dir, transform=None):
            self.image_folder = ImageFolder(root=root_dir,transform=transform)
            self.transform = transform

        def __len__(self):
            return max(len(self.image_folder),batch_size*1000)

        def __getitem__(self, idx):
            idx = idx % len(self.image_folder)
            image, label = self.image_folder[idx]
            return image
        
    #kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
    custom_datasett = CustomDataset(root_dir=rot_dir, transform=transform)
    dataloader = DataLoader(custom_datasett, batch_size=batch_size, shuffle=True)


    return dataloader,custom_datasett