from code.utils import preds_to_span_preds

import numpy as np
import torch
from sklearn import metrics


class Scorer:

    def __init__(self, datamodule):

        self.datamodule = datamodule
        self.is_setup = False

    def setup(self):

        self.dataset = self.datamodule.valid_dataset

        self.labels = []

        for essay_labels in self.dataset.data_labels:
            for _, _, label in essay_labels:
                self.labels.append(label)

        self.labels = np.array(self.labels)

    def merge_cut_preds(self, model_preds):

        index = 0
        preds_tmp = []
        text_indexes = self.dataset.text_indexes

        overlap = self.dataset.stride // 2

        while index < len(model_preds):

            text_index, _ = text_indexes[index]
            x_length = len(self.dataset.label_ids[text_index])

            preds = torch.zeros((x_length, 3))

            while index < len(model_preds):
                curr_text_index, start = text_indexes[index]

                if curr_text_index != text_index:
                    break

                curr_preds = model_preds[index]

                if start == 0:
                    length = min(len(preds), len(curr_preds))
                    preds[:length] = curr_preds[:length]
                elif start + len(curr_preds) > x_length:
                    preds[-len(curr_preds)+overlap:] = curr_preds[overlap:]
                else:
                    preds[start+overlap:start+len(curr_preds)] = curr_preds[overlap:]

                index += 1

            preds_tmp.append(preds)

        return preds_tmp

    def score(self, preds):

        if not self.is_setup:
            self.setup()
            self.is_setup = True

        preds = self.merge_cut_preds(preds)

        label_ids = self.dataset.label_ids
        span_labels = self.dataset.span_labels
        label_counts = self.dataset.label_counts

        preds, new_span_preds, _ = preds_to_span_preds(preds, label_ids, span_labels, label_counts)

        preds = preds.softmax(-1).detach().cpu()

        v = metrics.log_loss(
            new_span_preds.numpy(),
            preds.numpy(),
            labels=[0, 1, 2],
        )

        return v
