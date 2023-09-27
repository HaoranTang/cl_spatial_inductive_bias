import numpy as np
from torchvision import datasets, transforms
from .imagenetcls import ImageNetClsDataset
from PIL import Image

from io import BytesIO
import cv2

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def shotnoise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

def defocusblur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255

def jpeg(x, severity=1):
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)

    return x 

class ImageShuffler:
    # set seed=None to use default random state
    def __init__(self, mode, ny, nx, h=32, w=32, seed=42):
        assert h % ny == 0, 'h should be divisible by ny'
        assert w % nx == 0, 'w should be divisible by nx'
        self.mode = mode
        self.ny, self.nx = ny, nx
        self.h, self.w = h, w
        dy, dx = h // ny, w // nx
        p = np.empty([h, w], dtype=np.int32)
        r = np.arange(h * w).reshape(h, w)
        if mode == 'glo':
            q = np.random.RandomState(seed).permutation(ny * nx).reshape(ny, nx)
            for y in range(ny):
                for x in range(nx):
                    z = q[y, x]
                    v, u = z // nx, z % nx
                    p[y*dy:(y+1)*dy, x*dx:(x+1)*dx] = r[v*dy:(v+1)*dy, u*dx:(u+1)*dx]
        elif mode == 'loc':
            q = np.random.RandomState(seed).permutation(dy * dx)
            for y in range(ny):
                for x in range(nx):
                    v, u = y, x
                    p[y*dy:(y+1)*dy, x*dx:(x+1)*dx] = r[v*dy:(v+1)*dy, u*dx:(u+1)*dx].reshape(-1)[q].reshape(dy,dx)
        self.p = p.reshape(-1)
        self.p_inv = np.empty(h*w, dtype=np.int32)
        self.p_inv[self.p] = r.reshape(-1)

    # img: shape (h,w,c)
    def apply_one(self, img):
        s = img.shape
        assert len(s) == 3 and s[0] == self.h and s[1] == self.w
        return img.reshape(-1, s[-1])[self.p].reshape(s)
    
    # imgs: shape (n,h,w,c)
    def apply_batch(self, imgs):
        s = imgs.shape
        assert len(s) == 4
        if s[1] == self.h and s[2] == self.w:
            return imgs.reshape(s[0], -1, s[-1])[:, self.p].reshape(s)
        elif s[2] == self.h and s[3] == self.w:
            return imgs.reshape(s[0], s[1], -1)[:, :, self.p].reshape(s)
    
    def invert_one(self, img):
        s = img.shape
        return img.reshape(-1, s[-1])[self.p_inv].reshape(s)
    
    def invert_batch(self, imgs):
        s = imgs.shape
        return imgs.reshape(s[0], -1, s[-1])[:, self.p_inv].reshape(s)


def gamma_distortion_uint8(x, gamma):
    # print(x.dtype, x.shape, x.min(), x.max())  # uint8, (50000, 32, 32, 3) 0 255
    assert x.dtype == np.uint8
    return (255.*((x/255.)**gamma)).astype(np.uint8)


def corrupt_dataset(DatasetClass, corrupt):
    corrupt = corrupt.split('_')

    # if not corrupt or (corrupt.startswith('ori') and not resize):
    #     return DatasetClass
    # if not resize:
    if DatasetClass is datasets.CIFAR10 or DatasetClass is datasets.CIFAR100:
        image_size = 32

        class CorruptedDataset(DatasetClass):
            def __init__(self, *args, transform=None, **kwargs):
                super().__init__(*args, **kwargs)
                resize = transforms.Resize((image_size, image_size))
                # transform = transforms.Compose([transforms.ToTensor(),
                #                                 transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))])
                if corrupt[0] == 'glo' or corrupt[0] == 'loc':
                    shuf = ImageShuffler(corrupt[0], ny=int(corrupt[1]), nx=int(corrupt[1]), h=image_size, w=image_size)
                    def aug_func(x):
                        x = resize(x)
                        x = np.asarray(x)
                        x = shuf.apply_one(x)
                        x = Image.fromarray(x)
                        if transform is not None: x = transform(x)
                        return x
                elif corrupt[0] == 'randglo' or corrupt[0] == 'randloc':
                    def aug_func(x):
                        shuf = ImageShuffler(corrupt[0][-3:], ny=int(corrupt[1][-3:]), nx=int(corrupt[1][-3:]), h=image_size, w=image_size, seed=None)
                        x = resize(x)
                        x = np.asarray(x)
                        x = shuf.apply_one(x)
                        x = Image.fromarray(x)
                        if transform is not None: x = transform(x)
                        return x
                elif corrupt[0] == 'gam':
                    def aug_func(x):
                        x = resize(x)
                        x = np.asarray(x)
                        x = gamma_distortion_uint8(x, gamma=float(corrupt[1]))
                        x = Image.fromarray(x)
                        if transform is not None: x = transform(x)
                        return x
                elif corrupt[0] == 'ori':
                    def aug_func(x):
                        x = resize(x)
                        if transform is not None: x = transform(x)
                        return x
                else:
                    def aug_func(x):
                        x = resize(x)
                        x = eval(corrupt[0])(x, 3)
                        x = Image.fromarray(np.uint8(x))
                        if transform is not None: x = transform(x)
                        return x
                self.transform = aug_func

        return CorruptedDataset
    
    elif DatasetClass is datasets.STL10:
        image_size = 96
    elif DatasetClass is ImageNetClsDataset:
        image_size = 256

        class CorruptedDataset(DatasetClass):
            def __init__(self, *args, transform=None, **kwargs):
                super().__init__(*args, **kwargs)
                resize = transforms.Resize((image_size, image_size))
                if corrupt[0] == 'glo' or corrupt[0] == 'loc':
                    shuf = ImageShuffler(corrupt[0], ny=int(corrupt[1]), nx=int(corrupt[1]), h=image_size, w=image_size)
                    def aug_func(x):
                        x = resize(x)
                        x = np.asarray(x)
                        x = shuf.apply_one(x)
                        x = Image.fromarray(x)
                        if transform is not None: x = transform(x)
                        return x
                elif corrupt[0] == 'gam':
                    def aug_func(x):
                        x = resize(x)
                        x = np.asarray(x)
                        x = gamma_distortion_uint8(x, gamma=float(corrupt[1]))
                        x = Image.fromarray(x)
                        if transform is not None: x = transform(x)
                        return x
                else:
                    def aug_func(x):
                        x = resize(x)
                        x = eval(corrupt[0])(x, 3)
                        x = Image.fromarray(np.uint8(x))
                        if transform is not None: x = transform(x)
                        return x
                self.transform = aug_func

        return CorruptedDataset

    else:
        raise ValueError(f'unsupported {DatasetClass}')
    # else:
    #    image_size = resize

    class CorruptedDataset(DatasetClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # if resize:
            #     self.data = torch.nn.functional.interpolate(
            #         torch.as_tensor(self.data, dtype=torch.float32), [image_size, image_size], mode='bicubic').numpy()
            if corrupt[0] == 'glo' or corrupt[0] == 'loc':
                shuf = ImageShuffler(corrupt[0], ny=int(corrupt[1]), nx=int(corrupt[1]), h=image_size, w=image_size)
                self.data = shuf.apply_batch(self.data)
            elif corrupt[0] == 'gam':
                self.data = gamma_distortion_uint8(self.data, gamma=float(corrupt[1]))

    return CorruptedDataset
