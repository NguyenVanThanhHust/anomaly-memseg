from torchvision import datasets, transforms

class MVTec_Datasets(datasets):
    def __init__(self, data_folder):
        self.data_folder = data_folder
    
    def __len__(self):
        return 
    
    def __getitem__(self, idx):
        return 