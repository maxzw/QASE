"""Evaluation module"""
import wandb
import logging
from tqdm import tqdm
     
import numpy as np
from pandas import DataFrame
from collections import defaultdict
from typing import Mapping, Optional, Sequence

from torch.utils.data import DataLoader

from loader import QueryBatchInfo, QueryTargetInfo
from models import AnswerSpaceModel

logger = logging.getLogger(__name__)


class ClassificationReport:
    """Collects and aggregates classification reports during the whole training process"""
    def __init__(self):
        self.src = DataFrame()

    def flatten_dict(self, data: Mapping[str, Mapping[str, float]]) -> Mapping[str, float]:
        """
        Concatenates the keys in the nested input dict.

        Args:
            data (Mapping[str, Mapping[str, float]]): 
                See ClassificationData.finalize() for details.

        Returns:
            Dict to be included in the dataframe:
                dict { 
                    'structure_1_acc': float,
                    'structure_1_pre': float,
                    'structure_1_rec': float,
                    ...
                    'structure_n_acc': float,
                    'structure_n_pre': float,
                    'structure_n_rec': float,
                    'macro_acc': float,
                    ...
                    'weighted_rec': float,
                }
        """
        out_dict = {}
        for structure, metrics in data.items():
            for metric, value in metrics.items():
                out_dict[f"{structure}_{metric}"] = value
        return out_dict

    def include(self, data: dict, epoch: int) -> None:
        """Include the evaluation data in dataframe.

        Args:
            data (dict): See ClassificationData.finalize() for details.
            ref_index (dict): A reference index ('unit': index) used for plotting.
        """
        self.src = self.src.append({"epoch": epoch, **self.flatten_dict(data)}, ignore_index=True)


class ClassificationData:
    """Collects and aggregates classification results for a single evaluation run"""
    def __init__(self):
        self.src = defaultdict(lambda: defaultdict(float))

    def include(self, results) -> None:
        """Include new classification metrics in report (see classification_metrics)"""
        if not self.src:
            self.src = results
        else:
            for structure, metrics in results.items():
                for metric, values in metrics.items():
                    self.src[structure][metric] += values

    def finalize(self) -> Mapping[str, Mapping[str, float]]:
        """
        Calculate mean metrics per query structure, add global macro-,
        weighted-average and f1-score per metric and return resulting dictionary.

        Returns:
            dict {
                'structure_1': {
                    acc:    float
                    pre:    float
                    rec:    float
                    f1:     float
                },
                'structure_2': {
                    ...
                },
                'macro': {
                    acc:    float
                    pre:    float
                    rec:    float    
                    f1:     float
                },
                'weighted': {
                    acc:    float
                    pre:    float
                    rec:    float
                    f1:     float
                }
            }
        """
        if not self.src:
            return self.src

        global_accuracy_m = []
        global_precision_m = []
        global_recall_m = []

        global_accuracy_w = []
        global_precision_w = []
        global_recall_w = []

        for structure, metrics in self.src.items():
            # add raw values to weighted mean lists
            global_accuracy_w += metrics['acc']
            global_precision_w += metrics['pre']
            global_recall_w += metrics['rec']
            # add mean values to output dict
            for metric, values in metrics.items():
                self.src[structure][metric] = np.mean(values)
            # add structure-specific f1-score
            self.src[structure]['f1'] = 2*(
                (self.src[structure]['pre']*self.src[structure]['rec'])
                /(self.src[structure]['pre']+self.src[structure]['rec'])) \
                    if (self.src[structure]['pre']+self.src[structure]['rec']) > 0 else 0
            # add means to macro mean lists
            global_accuracy_m.append(metrics['acc'])
            global_precision_m.append(metrics['pre'])
            global_recall_m.append(metrics['rec'])

        self.src['macro'] = {}
        self.src['macro']['acc'] = np.mean(global_accuracy_m)
        self.src['macro']['pre'] = np.mean(global_precision_m)
        self.src['macro']['rec'] = np.mean(global_recall_m)
        self.src['macro']['f1'] = 2*(
            (self.src['macro']['pre']*self.src['macro']['rec'])
            /(self.src['macro']['pre']+self.src['macro']['rec'])) \
                if (self.src['macro']['pre']+self.src['macro']['rec']) > 0 else 0

        self.src['weighted'] = {}
        self.src['weighted']['acc'] = np.mean(global_accuracy_w)
        self.src['weighted']['pre'] = np.mean(global_precision_w)
        self.src['weighted']['rec'] = np.mean(global_recall_w)
        self.src['weighted']['f1'] = 2*(
             (self.src['weighted']['pre']*self.src['weighted']['rec'])
            /(self.src['weighted']['pre']+self.src['weighted']['rec'])) \
                if (self.src['weighted']['pre']+self.src['weighted']['rec']) > 0 else 0

        return dict(self.src)


def classification_metrics(
    preds: Sequence[Sequence[int]],
    trues: Sequence[Sequence[int]],
    modes: Sequence[str],
    nodes_per_mode: Mapping[str, Sequence[int]],
    q_types: Sequence[str],
    ) -> Mapping[str, Mapping[str, Sequence[float]]]:
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
    results = defaultdict(lambda: defaultdict(list))
    
    for pred_nodes, true_nodes, mode, q_type in zip(preds, trues, modes, q_types):
        
        tp = len([n for n in pred_nodes if n in true_nodes])
        tn = len([n for n in nodes_per_mode[mode] if (n not in pred_nodes) and (n not in true_nodes)])
        fp = len([n for n in pred_nodes if n not in true_nodes])
        fn = len([n for n in true_nodes if n not in pred_nodes])
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results[q_type]['acc'].append(accuracy)
        results[q_type]['pre'].append(precision)
        results[q_type]['rec'].append(recall)
    
    return dict(results)


def evaluate(
    model: AnswerSpaceModel,
    dataloader: DataLoader,
    epoch: Optional[int] = None
    ) -> Mapping[str, Mapping[str, float]]:
    """
    Evaluation function.

    Args:
        model (AnswerSpaceModel): The model.
        dataloader (DataLoader): Dataloader that contains the data to be evaluated.
        epoch (Optional[int], optional): The current epoch in which evaluation occurs.
            Indicates that we track answer space size. Defaults to None.

    Returns:
        Mapping[str, Mapping[str, float]]: Dictionary with results (see ClassificationData.finalize).
    """
    
    # Put the model in eval mode
    model.eval()

    # Track classification metrics during eval run
    classif_data = ClassificationData()

    # Track and log average answer set size to WandB as proxy for answer space size
    if epoch is not None:
        answer_sizes = np.empty(len(dataloader))

    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for batch_id, (x_info, y_info) in enumerate(tqdm(dataloader, desc="Evaluate", unit="batch", position=1, leave=False)):
        
        hyp = model(x_info)
        preds = model.predict(hyp)
        results = classification_metrics(
            preds,
            y_info.target_nodes,
            y_info.pos_modes,
            model.nodes_per_mode,
            y_info.q_types,
            )
        if epoch == 20:
            np.save("preds.npy", preds, allow_pickle=True, fix_imports=True)
            exit()
        classif_data.include(results)

        # Add the average of the fractions of predicted answers respective to the total number of typed entities
        if epoch is not None:
            answer_sizes[batch_id] = np.mean([len(a) for a in preds])

    # Log the average answer size from all batches
    if epoch is not None:
        wandb.log({"val": {"mean_answer_size": np.mean(answer_sizes), "epoch_id": epoch}})

    return classif_data.finalize()
