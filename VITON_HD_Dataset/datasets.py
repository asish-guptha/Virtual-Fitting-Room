import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VITONDataset(Dataset):
    def __init__(self, dataset_dir, dataset_list):
        self.dataset_dir = dataset_dir
        self.samples = self.load_samples(dataset_list)

        # ✅ Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 192)),  # Adjust based on model input size
            transforms.ToTensor(),
        ])

    def load_samples(self, dataset_list):
        """ Load image and clothing pairs from dataset_list file. """
        with open(dataset_list, 'r') as f:
            lines = f.readlines()
        return [line.strip().split() for line in lines]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ Load and return an image-clothing pair as tensors. """
        image_name, cloth_name = self.samples[idx]
        
        image_path = os.path.join(self.dataset_dir, "image", image_name)
        cloth_path = os.path.join(self.dataset_dir, "cloth", cloth_name)

        # ✅ Load images
        image = Image.open(image_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")

        # ✅ Apply transformations
        image = self.transform(image)
        cloth = self.transform(cloth)

        return {"image": image, "cloth": cloth, "image_name": image_name}

class VITONDataLoader():
    def __init__(self, opt, dataset):
        self.dataset = dataset
        self.batch_size = getattr(opt, "batch_size", 1)  # ✅ Ensure batch_size exists
        self.shuffle = getattr(opt, "shuffle", False)  # ✅ Default shuffle to False
        self.num_workers = getattr(opt, "workers", 0)  # ✅ Default workers to 0

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def load_data(self):
        return self.dataloader
