from torchvision import transforms
import numpy as np

WID = 112
HEI = 112

transform_rgb_residual = transforms.Compose([
    transforms.Resize(np.random.randint(WID, 224)),
    transforms.ColorJitter(1.0, 1.0, 1.0, 0.25),
    transforms.RandomAffine(15),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(WID, HEI)),
    transforms.ToTensor(),
])

transform_mv = transforms.Compose([
    transforms.Resize(np.random.randint(WID, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(WID, HEI)),
    transforms.ToTensor(),
])


transform_infer = transforms.Compose([
    transforms.Resize(np.random.randint(WID, 224)),
    transforms.CenterCrop(size=(WID, HEI)),
    transforms.ToTensor(),
])
