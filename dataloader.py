import numpy as np
import torch
import torchaudio

from config import FS


# def test():
#     import numpy as np
#     x = torch.tensor(np.arange(100))
#     y = torch.tensor(np.arange(100))
#     dataset = AudioDataset([x, y], split_length=9)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=1)
#     for _ in range(3):
#         for i, data in enumerate(loader):
#             print(str(i)+' '*4+str(data))


class AudioDataset(torch.utils.data.TensorDataset):
    def __init__(self, tensors, split_length, is_training):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        tensors = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in tensors]
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        split_length = int(np.ceil(tensors[0].size(0) / (tensors[0].size(0) // split_length)))
        print('split length: %i' % split_length)

        if tensors[0].size(0) % split_length != 0:
            zeros = torch.zeros(split_length-(tensors[0].size(0) % split_length))
            tensors = [torch.cat([t, zeros], dim=0) for t in tensors]

        self.tensors = tensors
        self.split_length = split_length
        self.is_training = is_training

    def __getitem__(self, index):
        if self.is_training:
            if index == 0:
                shift = torch.distributions.uniform.Uniform(torch.tensor([0.]),
                                                            torch.tensor([float(self.split_length/2+1)]))
            elif index == self.__len__()-1:
                shift = torch.distributions.uniform.Uniform(torch.tensor([-float(self.split_length/2)]),
                                                            torch.tensor([0.]))
            else:
                shift = torch.distributions.uniform.Uniform(torch.tensor([-float(self.split_length/2)]),
                                                            torch.tensor([float(self.split_length/2+1)]))
            shift = shift.sample().type(torch.int)
            sample = [tensor[index*self.split_length+shift:(index+1)*self.split_length+shift]
                      for tensor in self.tensors]
        else:
            sample = [tensor[index*self.split_length:(index+1)*self.split_length] for tensor in self.tensors]
        sample = [tensor.unsqueeze(0) for tensor in sample]
        return tuple(sample)

    def __len__(self):
        return self.tensors[0].size(0)//self.split_length
