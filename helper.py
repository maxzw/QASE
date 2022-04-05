import logging
from tqdm import tqdm
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


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Source: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """    
    def __init__(self, patience: int = 5, min_delta: int = 0):
        """_summary_

        Args:
            patience (int, optional): how many epochs to wait before stopping when loss is
               not improving. Defaults to 5.
            min_delta (int, optional): minimum difference between new loss and old loss for
               new loss to be considered as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            logging.info(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logging.info('INFO: Early stopping')
                self.early_stop = True

    def __repr__(self):
        return f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta})"