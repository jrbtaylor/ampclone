import torch

BASE_DIR = '/home/jtaylor/cloner/'  # 'C:/Users/Jason/Documents/cloner/'
IR_FILE = BASE_DIR + 'IR/OH_212_ORNG_V30+M25_PROG-05_32bit_44100.wav'
FS = 44100
SPLIT_LENGTH = 5e5  # reasonable limit for gpu training w/ FIR filters
VAL_SPLIT_LENGTH = 1e6

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')
