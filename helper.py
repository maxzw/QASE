import sys
if 'google.colab' in sys.modules:
    print("Running in Colab")
    from tqdm.notebook import tqdm
else:
    print("NOT running in Colab")
    from tqdm import tqdm

import logging
from datetime import datetime

import torch
from torch.nn.init import xavier_normal_
from torch.nn import Parameter


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

def create_logger(dataset: str):
    dt = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    filename = f"./results/{dataset}/logs/{dt}.txt"
    open(filename, "x").close()
    logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(name)5s | %(levelname)5s | %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S',
                handlers=[
                    logging.FileHandler(filename),
                    TqdmLoggingHandler()
                ])

def get_param(shape):
    param = Parameter(torch.Tensor(*shape)); 	
    xavier_normal_(param.data)
    return param
