import torch 
from torchvision import transforms 
from torchvision.transforms import v2
from torchvision .transforms.functional import InterpolationMode
from torch.utils.data import default_collate


### SET Transforms  for Training and Testing ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def train_transforms(img_size = (224,224),
                     image_mean = IMAGENET_MEAN,
                     image_std = IMAGENET_STD,
                     hflip_probability = 0.5,
                     interpolation = InterpolationMode.BILINEAR,
                     random_aug_magnitude = 9):

    transformation_chain = []

    transformation_chain.append(v2.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=interpolation, antialias=None)) 
    
    if hflip_probability > 0:
        transformation_chain.append(v2.RandomHorizontalFlip(p=hflip_probability))

    if random_aug_magnitude > 0:
        transformation_chain.append(v2.RandAugment(magnitude=random_aug_magnitude, interpolation=interpolation))

    
    transformation_chain.append(v2.PILToTensor())

    transformation_chain.append(v2.ToDtype(torch.float32, scale=True))

    transformation_chain.append(v2.Normalize(mean=image_mean, std=image_std))

    return transforms.Compose(transformation_chain)


def eval_transforms(img_size = (224,224),
                    resize_size = (256,256),
                    image_mean = IMAGENET_MEAN,
                    image_std = IMAGENET_STD,
                    interpolation = InterpolationMode.BILINEAR):

    transformations = transforms.Compose(
        [
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            v2.CenterCrop(img_size),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std),
    ])               
    
    return transformations


def mixup_cutmix_collate_fn(mixup_alpha = 0.2,
                            cutmix_alpha = 1.0,
                            num_classes = 1000):

    mix_cut_transform = None

    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(v2.Mixup(alpha = mixup_alpha, num_classes = num_classes))

    if cutmix_alpha > 0:
        mixup_cutmix.append(v2.CutMix(alpha = cutmix_alpha, num_classes = num_classes))

    if len(mixup_cutmix) > 0:
        mix_cut_transform = v2.RandomChoice(mixup_cutmix)

    def collate_fn(batch):

        collated = default_collate(batch)

        if mix_cut_transform is not None:
            collated = mix_cut_transform(collated)

    return collate_fn        
    


def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.shape[0]

    if target.ndim == 2:
        target = target.max(dim=-1)[1]   # class label with highest proportion will be actual target

    values, pred = output.topk(maxk, dim = -1, largest=True, sorted=True)
    
    pred = pred.transpose(0,1)
    correct = (pred == target)

    accs = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)   # sum all correct predictio
        accs.append((correct_k/ batch_size))

    if len(accs) == 1:
        return accs[0]   
    else:
        return accs       



    

if __name__ == "__main__":


    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    train_transform = train_transforms()
    dataset = ImageFolder("/mnt/datadrive/data/dogsvscats", transform=train_transform)
    
    collate_fn = mixup_cutmix_collate_fn(num_classes=1000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    samples, labels = next(iter(loader))
    rand_logits = torch.randn(32, 1000)

    accuracy(rand_logits, labels)


