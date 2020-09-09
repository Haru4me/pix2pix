from PIL import Image
from pathlib import Path
from torchvision import transforms


"""
Load Dataset
"""


class Data:

    def __init__(self,path,swap=False):

        if swap == False:
          
            self.A_path = sorted(Path(path+"A").rglob('*.jpg'))
            self.B_path = sorted(Path(path+"B").rglob('*.jpg'))

        else:

            self.A_path = sorted(Path(path+"B").rglob('*.jpg'))
            self.B_path = sorted(Path(path+"A").rglob('*.jpg'))
        
            
    def __getitem__(self, index):
        
        A,B =  self.A_path[index % len(self.A_path)], self.B_path[index% len(self.B_path)]
        trans = transforms.Compose([transforms.Resize(286),  
                             transforms.CenterCrop(256),
                             transforms.ToTensor()])  
        image_A = trans(Image.open(A))
        image_B = trans(Image.open(B))

        return {"A": image_A, "B": image_B}

    def __len__(self):
        return max(len(self.A_path), len(self.B_path))