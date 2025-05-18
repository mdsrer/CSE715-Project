import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms

class GTZANDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.genres = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.file_paths = []
        self.labels = []
        
        for genre in self.genres:
            genre_dir = os.path.join(root_dir, genre)
            files = [os.path.join(genre_dir, f) for f in os.listdir(genre_dir) 
                    if f.endswith('.png') or f.endswith('.jpg')]
            self.file_paths.extend(files)
            self.labels.extend([genre] * len(files))
        
        if train is not None:
            indices = list(range(len(self.file_paths)))
            split = int(0.8 * len(indices))
            train_indices = indices[:split]
            test_indices = indices[split:]
            
            if train:
                self.file_paths = [self.file_paths[i] for i in train_indices]
                self.labels = [self.labels[i] for i in train_indices]
            else:
                self.file_paths = [self.file_paths[i] for i in test_indices]
                self.labels = [self.labels[i] for i in test_indices]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        label = self.labels[idx]
        return image, label
    
    def get_num_channels(self):
        sample_img, _ = self.__getitem__(0)
        return sample_img.shape[0]