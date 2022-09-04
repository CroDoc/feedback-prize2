import numpy as np
import torch

discourse_type_to_label = {
    'O': 0,
    'B-Claim': 1,
    'I-Claim': 2,
    'B-Evidence': 3,
    'I-Evidence': 4,
    'B-Lead': 5,
    'I-Lead': 6,
    'B-Position': 7,
    'I-Position': 8,
    'B-Counterclaim': 9,
    'I-Counterclaim': 10,
    'B-Rebuttal': 11,
    'I-Rebuttal': 12,
    'B-Concluding Statement': 13,
    'I-Concluding Statement': 14,
}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9)
    return sum_embeddings / sum_mask

def max_pooling(model_output, attention_mask):
    return torch.max(model_output[attention_mask], 0)[0]

def fast_mean_pooling(output_view, div):
    return torch.sum(output_view, 0) / div

def preds_to_span_preds(output, label_ids, span_labels, label_counts):

    preds, new_span_labels, loss_divs = [], [], []

    for row_preds, row_label_ids, row_span_label, row_label_counts in zip(output, label_ids, span_labels, label_counts):

        start_dict = {}
        end_dict = {}

        for idx, label_id in enumerate([x.item() for x in row_label_ids]):

            if label_id == -1:
                continue

            if start_dict.get(label_id, -1) == -1:
                start_dict[label_id] = idx

            end_dict[label_id] = idx + 1

        sorted_labels = sorted(start_dict.keys())

        for label_id in sorted_labels:

            start, end = start_dict[label_id], end_dict[label_id]

            preds.append(fast_mean_pooling(row_preds[start:end], end-start))
            #preds.append(mean_pooling(row_preds, row_label_ids==label_id))
            #preds.append(max_pooling(row_preds, row_label_ids==label_id))
            new_span_labels.append(row_span_label[label_id])
            #loss_divs.append(row_label_counts[label_id] / sum(row_label_ids==label_id))

    if len(preds) > 0:
        preds = torch.stack(preds)

    new_span_labels = torch.tensor(new_span_labels)
    #loss_divs = torch.tensor(loss_divs)

    return preds, new_span_labels, None #loss_divs
