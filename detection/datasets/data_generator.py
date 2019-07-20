import numpy as np

class DataGenerator(object):
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
    
    def __call__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for img_idx in indices:
            data = self.dataset[img_idx]
            yield data
