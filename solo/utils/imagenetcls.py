import torch.utils.data
from PIL import Image
from io import BytesIO
import os

class ImageNetClsDataset(torch.utils.data.Dataset):
    '''
    More efficient dataset interface with images in binary format
    
    e.g.
      img_file:   imagenet2012/train.binary
        concatenated raw JPG/PNG/... files
      label_file: imagenet2012/train.binary.label
        lines of tab separated (name, label_id, offset, size) tuples
    '''
    def __init__(self, root='datasets', split='train', transform=None, **kwargs):
        img_file = os.path.join(root, f'{split}.binary')
        label_file = os.path.join(root, f'{split}.binary.label')

        download_data = False
        if download_data:
            extra_desc.remove('down')

            import fcntl
            def acquireLock():
                ''' acquire exclusive lock file access '''
                locked_file_descriptor = open('/tmp/lockfile.LOCK', 'w+')
                fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
                return locked_file_descriptor

            def releaseLock(locked_file_descriptor):
                ''' release exclusive lock file access '''
                locked_file_descriptor.close()

            acquireLock()

            img_file_local = '/dev/shm/databin'
            if not os.path.isfile(img_file_local):
                print (f'copying file to {img_file_local} ...')
                from shutil import copyfile
                copyfile(img_file, img_file_local)

            releaseLock()

        # support multiprocess
        self.fname = img_file
        self.pid = -1
        # self.ensure_file_open()

        with open(label_file, 'r') as f:
            self.labels = [[int(x) for x in _.split('\t')[1:6]]
                    for _ in f.read().splitlines()]
        self._num_classes = None
        # self.num_classes = max([_[0] for _ in self.labels]) + 1
        # labels = [_[0] for _ in self.labels]
        # from collections import Counter
        # cnt = Counter(labels)
        # print (f'load {label_file}, len={len(self.labels)}, cnt={cnt}')
        print (f'load {label_file}, len={len(self.labels)}')
        self.transform = transform

    def ensure_file_open(self):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.f = open(self.fname, 'rb')

    def __getitem__(self, i):
        label, offset, size = self.labels[i][:3]
        self.ensure_file_open()
        self.f.seek(offset)
        b = BytesIO(self.f.read(size))
        img = Image.open(b).convert('RGB')
        img.load()
        b.close()
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

    def get_img_info(self, i):
        w, h = self.labels[i][3:5]
        return {'height': h, 'width': w}

    @property
    def num_classes(self):
        if self._num_classes is None:
            self._num_classes = max([_[0] for _ in self.labels]) + 1
        return self._num_classes
