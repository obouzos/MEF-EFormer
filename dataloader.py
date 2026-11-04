# dataloader.py
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


class MEFDataset(Dataset):
    def __init__(self, inputs_dir, targets_dir):
        self.under_dir = Path(inputs_dir) / "under"
        self.over_dir = Path(inputs_dir) / "over"
        self.targets_dir = Path(targets_dir)
        self.image_names = sorted(path.name for path in self.under_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def _load_image(path):
        with Image.open(path) as image:
            return pil_to_tensor(image.convert("RGB")).float().div_(255.0)

    def __getitem__(self, index):
        name = self.image_names[index]
        under = self._load_image(self.under_dir / name)
        over = self._load_image(self.over_dir / name)
        target = self._load_image(self.targets_dir / name)
        return under, over, target

def make_train_loader(inputs_dir, targets_dir, batch_size=1, num_workers=4, pin_memory=True):
    dataset = MEFDataset(inputs_dir, targets_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)