"""Evaluation functions"""
from pandas import DataFrame
from tqdm import tqdm
from torch.utils.data import DataLoader

from loader import QueryBatchInfo, QueryTargetInfo
from models import AnswerSpaceModel


class ClassificationReport:
    """Collects and aggregates classification results for a single evaluation run"""
    def __init__(self):
        self.src = dict()

    def include(self, results):
        """Include new classification metrics in report"""
        pass

    def finalize(self):
        """
        Calculate mean metrics per query structure, and add
        global mean and weighed statistics.
        """
        pass


class ClassificationData:
    """Collects and aggregates classification reports during the whole training process"""
    def __init__(self):
        self.src = DataFrame()

    def include(self, results):
        """Include new classification report in dataframe"""
        pass


def classification_metrics(
    preds,
    y_info
    ):
    """
    Return classification results:
    dict {
        'structure_1': {
            accuracy:   [q1, q2, q3, ...]
            precision:  [q1, q2, q3, ...]
            recall:     [q1, q2, q3, ...]
        }
        'structure_2': {
            ...
        }
    }
    """


def evaluate(
    model: AnswerSpaceModel,
    dataloader: DataLoader,
    ):
    
    # put the model in eval mode
    model.eval()

    report = ClassificationReport()

    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for x_info, y_info in tqdm(dataloader, desc="Evaluation", unit="batch", position=1, leave=False):
        
        hyp = model(x_info)
        preds = model.predict(hyp)
        results = classification_metrics(preds, y_info)
        report.include(results)

    report.finalize()
    return report.src

