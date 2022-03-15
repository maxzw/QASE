"""Evaluation functions"""
from collections import defaultdict
from typing import Mapping, Sequence
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from torch.utils.data import DataLoader

from loader import QueryBatchInfo, QueryTargetInfo
from models import AnswerSpaceModel


class ClassificationReport:
    """Collects and aggregates classification reports during the whole training process"""
    def __init__(self):
        self.src = DataFrame()

    def include(self, results):
        """Include new classification report in dataframe"""
        pass


class ClassificationData:
    """Collects and aggregates classification results for a single evaluation run"""
    def __init__(self):
        self.src = None

    def include(self, results):
        """Include new classification metrics in report (see classification_metrics)"""
        if self.src == None:
            self.src = results
        else:
            for structure, metrics in results.items():
                for metric, values in metrics.items():
                    self.src[structure][metric] += values

    def finalize(self):
        """
        Calculate mean metrics per query structure, and add
        global regular and weighed mean per metric.

        Returns:
            dict {
                'structure_1': {
                    acc:    float
                    pre:    float
                    rec:    float
                },
                'structure_2': {
                    ...
                },
                'mean': {
                    acc:    float
                    pre:    float
                    rec:    float                
                },
                'weighed': {
                    acc:    float
                    pre:    float
                    rec:    float
                }
            }
        """
        global_accuracy_m = []
        global_precision_m = []
        global_recall_m = []

        global_accuracy_w = []
        global_precision_w = []
        global_recall_w = []

        for structure, metrics in self.src.items():
            # add raw values to weighed mean lists
            global_accuracy_w += metrics['acc']
            global_precision_w += metrics['pre']
            global_recall_w += metrics['rec']
            for metric, values in metrics.items():
                self.src[structure][metric] = np.mean(values)
            # add means to regular mean lists
            global_accuracy_m += metrics['acc']
            global_precision_m += metrics['pre']
            global_recall_m += metrics['rec']
    
        self.src['mean']['acc'] = np.mean(global_accuracy_m)
        self.src['mean']['pre'] = np.mean(global_precision_m)
        self.src['mean']['rec'] = np.mean(global_recall_m)

        self.src['weighed']['acc'] = np.mean(global_accuracy_w)
        self.src['weighed']['pre'] = np.mean(global_precision_w)
        self.src['weighed']['rec'] = np.mean(global_recall_w)

        return self.src


def classification_metrics(
    preds: Sequence[Sequence[int]],
    trues: Sequence[Sequence[int]],
    modes: Sequence[str],
    nodes_per_mode: Mapping[str, Sequence[int]],
    q_types: Sequence[str],
    ):
    """
    Calculates classification statistics for a batch of queries.

    Args:
        preds (Sequence[Sequence[int]]):
            The predicted entity IDs that are the answer to the query.
        trues (Sequence[Sequence[int]]):
            The true entity IDs that are the answer to the query.
        modes (Sequence[str]):
            The entity type/mode that belongs to the target entity of the query.
        nodes_per_mode (Mapping[str, Sequence[int]]):
            The entity IDs that belong to a particular entity type.
        q_types (Sequence[str]):
            The query structures: ['1-chain', '2-chain', ..., '3-inter_chain'].

    Returns:
        dict {
            'structure_1': {
                acc:    [q1, q2, q3, ...]
                pre:    [q1, q2, q3, ...]
                rec:    [q1, q2, q3, ...]
            }
            'structure_2': {
                ...
            }
        }
    """
    results = defaultdict(defaultdict(list))
    
    for pred_nodes, true_nodes, mode, q_type in zip(preds, trues, modes, q_types):
        
        tp = len([n for n in pred_nodes if n in true_nodes])
        tn = len(nodes_per_mode[mode]) - len(set(pred_nodes + true_nodes))
        fp = len([n for n in pred_nodes if n not in true_nodes])
        fn = len([n for n in true_nodes if n in pred_nodes])
        
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        results[q_type]['acc'] += [accuracy]
        results[q_type]['pre'] += [precision]
        results[q_type]['rec'] += [recall]
    
    return results


def evaluate(
    model: AnswerSpaceModel,
    dataloader: DataLoader
    ):
    
    # put the model in eval mode
    model.eval()

    classif_data = ClassificationData()

    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for x_info, y_info in tqdm(dataloader, desc="Evaluation", unit="batch", position=1, leave=False):
        
        hyp = model(x_info)
        preds = model.predict(hyp, y_info.pos_modes)
        results = classification_metrics(
            preds,
            y_info.target_nodes,
            y_info.pos_modes,
            model.nodes_per_mode,
            y_info.q_types,
            )
        classif_data.include(results)
    
    return classif_data.finalize()
