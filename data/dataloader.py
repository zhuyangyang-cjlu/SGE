import numpy as np
import torch
import torch.utils.data

# class SharedClassSampler:
#     def __init__(self, n_classes, n_cls, n_batch, fix_seed=True):
#         self.n_classes = n_classes
#         self.n_cls = n_cls
#         self.n_batch = n_batch
#         self.fix_seed = fix_seed
#         self.cached_classes = None
#         if self.fix_seed:
#             np.random.seed(0)
#             self.cached_classes = []
#             for _ in range(self.n_batch):
#                 classes = np.random.choice(range(self.n_classes), self.n_cls, replace=False)
#                 self.cached_classes.append(classes)
#             self.cached_classes = np.array(self.cached_classes)

#     def get_classes(self):
#         if self.fix_seed:
#             return self.cached_classes
#         else:
#             classes = []
#             for _ in range(self.n_batch):
#                 class_sample = np.random.choice(range(self.n_classes), self.n_cls, replace=False)
#                 classes.append(class_sample)
#             return np.array(classes)



# class EpisodeSampler:
#     def __init__(self, label, n_batch, n_cls, n_per, shared_class_sampler, fix_seed=True):
#         self.n_batch = n_batch
#         self.n_cls = n_cls
#         self.n_per = n_per
#         self.fix_seed = fix_seed
#         self.shared_class_sampler = shared_class_sampler

#         label = np.array(label)
#         self.m_ind = []
#         for i in range(max(label) + 1):
#             ind = np.argwhere(label == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind.append(ind)

#         if self.fix_seed:
#             np.random.seed(0)
#             self.cached_batches = []
#             for i in range(self.n_batch):
#                 classes = self.shared_class_sampler.get_classes()[i]
#                 batch = self.generate_batch(classes)
#                 self.cached_batches.append(batch)
#             self.cached_batches = torch.stack(self.cached_batches)

#     def generate_batch(self, classes):
#         batch = []
#         for c in classes:
#             l = self.m_ind[c]
#             pos = np.random.choice(range(len(l)), self.n_per, False)
#             batch.append(l[pos])
#         return torch.stack(batch).reshape(-1)

#     def __len__(self):
#         return self.n_batch

#     def __iter__(self):
#         if self.fix_seed:
#             for batch in self.cached_batches:
#                 yield batch
#         else:
#             for i in range(self.n_batch):
#                 classes = self.shared_class_sampler.get_classes()[i]
#                 batch = self.generate_batch(classes)
#                 yield batch

# 定义一个EpisodeSampler类，用于采样一个batch的样本
# 参数：label：标签；n_batch：batch的数量；n_cls：每个batch中类别的数量；n_per：每个类别中样本的数量；fix_seed：是否固定随机种子
class EpisodeSampler:
    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed
        self.label = np.array(label)
        self.m_ind = [torch.from_numpy(np.argwhere(label == i).reshape(-1)) for i in range(max(label) + 1)]

        self.cached_batches = None
        if self.fix_seed:
            np.random.seed(0)
            self.cached_batches = []
            for _ in range(self.n_batch):
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                batch = self.generate_batch(classes)
                self.cached_batches.append(batch)
            self.cached_batches = torch.stack(self.cached_batches)

    def generate_batch(self, classes):
        batch = []
        for c in classes:
            l = self.m_ind[c]
            pos = np.random.choice(len(l), self.n_per, False)
            batch.append(l[pos])
        return torch.stack(batch).reshape(-1)

    def __iter__(self):
        if self.fix_seed:
            for batch in self.cached_batches:
                yield batch
        else:
            for _ in range(self.n_batch):
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, replace=False)
                batch = self.generate_batch(classes)
                yield batch

    def __len__(self):
        return self.n_batch
    
class TESTEpisodeSampler:
    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        if self.fix_seed:
            np.random.seed(0)
            self.cached_batches = []
            for i in range(self.n_batch):
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, True)
                    batch.append(l[pos])
                batch = torch.stack(batch).reshape(-1)
                self.cached_batches.append(batch)

            self.cached_batches = torch.stack(self.cached_batches)
        
            np.random.seed(0)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            if self.fix_seed:
                batch = self.cached_batches[i_batch]
            else:
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, False)
                    batch.append(l[pos])
                batch = torch.stack(batch).reshape(-1)
                

            yield batch

class RepeatSampler:
    def __init__(self, dataset, batch_size, repeat):
        self.batch_size = batch_size//repeat
        self.repeat = repeat
        self.sampler = torch.utils.data.RandomSampler(dataset)
        self.drop_last = True

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # 确保batch_size能够被repeat整除
                if len(batch) * self.repeat <= len(self.sampler):
                    batch = batch * self.repeat
                    yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class MultiTrans:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        out = []
        for trans in self.trans:
            out.append(trans(x))
        return out


import numpy as np
import torch

class view_EpisodeSampler:
    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        if self.fix_seed:
            np.random.seed(0)
            self.cached_batches = []
            for i in range(self.n_batch):
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.arange(len(l)) % self.n_per  # 循环选择样本
                    batch.append(l[pos][:self.n_per])  # 确保每个类别的样本数量一致
                batch = torch.stack(batch).reshape(-1)
                self.cached_batches.append(batch)
                print(i)
                
            self.cached_batches = torch.stack(self.cached_batches)
            np.random.seed(0)

    def generate_batch(self, classes):
        batch = []
        for c in classes:
            l = self.m_ind[c]
            pos = np.arange(len(l)) % self.n_per  
            batch.append(l[pos][:self.n_per]) 
        return torch.stack(batch).reshape(-1)

    def __iter__(self):
        if self.fix_seed:
            for batch in self.cached_batches:
                yield batch
        else:
            for _ in range(self.n_batch):
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                batch = self.generate_batch(classes)
                yield batch

    def __len__(self):
        return self.n_batch

