import torchvision.transforms as T
import torch.nn.functional as F


class CustomRandomCrop(T.RandomCrop):
    def __init__(self, size=84, padding=4, **kwargs):
        super().__init__(size, **kwargs)
        self.padding = padding

    def __call__(self, img):
        # first pad image by 4 pixels on each side
        img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        # crop to original size
        return super().__call__(img)

 
def make_augmentations(aug_kwargs=None):
    if aug_kwargs is None:
        aug_kwargs = {}
    aug_kwargs = aug_kwargs.copy() 
    kind = aug_kwargs.pop("kind", "crop_rotate")
    p_aug = aug_kwargs.get("p_aug", 0.5)
    if kind == "crop": 
        return T.RandomApply([CustomRandomCrop(**aug_kwargs)], p=p_aug)
    elif kind == "rotate": 
        degrees = aug_kwargs.pop("degrees", 30)
        return T.RandomApply([T.RandomRotation(degrees=degrees, **aug_kwargs)], p=p_aug)
    elif kind == "crop_rotate": 
        degrees = aug_kwargs.pop("degrees", 30)
        return T.Compose([
            T.RandomApply([CustomRandomCrop(**aug_kwargs)], p=p_aug),
            T.RandomApply([T.RandomRotation(degrees=degrees, **aug_kwargs)], p=p_aug)
        ])
    raise ValueError(f"Unknown augmentation kind: {kind}")
