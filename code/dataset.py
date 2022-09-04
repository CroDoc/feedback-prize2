import random
from code.utils import discourse_type_to_label

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, cfg, validation=False):

        self.tokenizer = tokenizer
        self.max_length = cfg['max_length']
        self.validation = validation
        self.cfg = cfg

        self.data_labels = df['labels'].values.tolist()
        self.stride = cfg['stride']

        self.mask = 0.0
        if 'mask' in cfg:
            self.mask = cfg['mask']

        if validation == True:
            self.x, self.span_labels, self.label_ids, self.label_counts = [], [], [], []
            self.x_cut, self.span_labels_cut, self.label_ids_cut, self.label_counts_cut = [], [], [], []
            self.text_indexes = []

            text_index = 0

            for idx in range(len(self.data_labels)):
                x, span_labels, label_ids, label_counts = self.make_item(idx)

                self.x.append(x)
                self.span_labels.append(span_labels)
                self.label_ids.append(label_ids)
                self.label_counts.append(label_counts)

                start = 0
                total_tokens = len(label_ids)

                break_bool = False

                while start < total_tokens and not break_bool:

                    if start + self.max_length > total_tokens:
                        start = max(0, total_tokens - self.max_length)
                        break_bool = True

                    x_cut, label_ids_cut = self.get_cut_item(x, label_ids, start)

                    self.x_cut.append(x_cut)
                    self.span_labels_cut.append(span_labels)
                    self.label_ids_cut.append(label_ids_cut)
                    self.label_counts_cut.append(label_counts)

                    self.text_indexes.append((text_index, start))

                    start += self.stride

                text_index += 1

    def get_cut_element(self, tokenized_element, start, length):
        return tokenized_element[start:start+length].clone()

    def get_cut_item(self, tokenized, label_ids, start):

        cut_length = min(self.max_length, len(label_ids))

        new_tokenized = {}

        for k in tokenized:
            new_tokenized[k] = self.get_cut_element(tokenized[k], start, cut_length)

        if label_ids is not None:
            label_ids = self.get_cut_element(label_ids, start, cut_length)

        return new_tokenized, label_ids

    def __len__(self):
        if self.validation:
            return len(self.x_cut)
        else:
            return len(self.data_labels)

    def add_masking(self, x, label_ids):
        mask_id = self.tokenizer.mask_token_id

        input_len = len(x['input_ids'])
        random_value = random.random() * self.mask
        indices = random.sample(range(input_len), int(input_len * random_value))

        for idx in indices:
            #old_token_id = x['input_ids'][idx]

            if label_ids[idx] != -1:
            #if old_token_id != self.tokenizer.cls_token_id and old_token_id != self.tokenizer.sep_token_id:
                x['input_ids'][idx] = mask_id

    def make_item(self, idx):

        data_labels = self.data_labels[idx]

        span_labels = []
        label_ids = [-1]
        input_ids = [self.tokenizer.cls_token_id]

        for idx, (discourse_text, discourse_type, score) in enumerate(data_labels):

            text_input_ids = self.tokenizer.encode(
                discourse_type.lower() + ' : ' + discourse_text,
                add_special_tokens = False,
            )

            span_labels.append(score)
            label_ids += [idx] * len(text_input_ids) + [-1]
            input_ids += text_input_ids + [self.tokenizer.sep_token_id]

        attention_mask = [1] * len(input_ids)

        tokenized = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v, dtype=torch.long)

        span_labels = torch.tensor(span_labels, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        label_counts = []

        for idx in range(len(span_labels)):
            label_counts.append(sum(label_ids == idx))

        label_counts = torch.tensor(label_counts, dtype=torch.long)

        return tokenized, span_labels, label_ids, label_counts

    def __getitem__(self, idx):

        if self.validation:
            return self.x_cut[idx], self.span_labels_cut[idx], self.label_ids_cut[idx], self.label_counts_cut[idx]

        x, span_labels, label_ids, label_counts = self.make_item(idx)

        random_value = random.random()

        min_start = 0
        max_start = max(0, len(label_ids) - self.max_length)

        if len(label_ids) <= self.max_length:
            start = min_start
        elif len(label_ids) > self.max_length and len(label_ids) <= 2 * self.max_length:
            if random_value < 0.5:
                start = min_start
            else:
                start = max_start
        else:
            if random_value < 0.25:
                start = min_start
            elif random_value < 0.25:
                start = max_start
            else:
                start = random.randint(min_start, max_start)

        start = 0

        x, label_ids = self.get_cut_item(x, label_ids, start)

        if start > 0:

            pos = 0

            while pos < self.stride // 4:
                label_ids[pos] = -1
                pos += 1

            if label_ids[pos] != -1:
                li = label_ids[pos].item()

                while pos < len(label_ids) and label_ids[pos] == li:
                    label_ids[pos] = -1
                    pos += 1

        pos = len(label_ids) - 1
        if label_ids[pos] != -1:
            li = label_ids[pos].item()

            while pos > 0 and label_ids[pos] == li:
                label_ids[pos] = -1
                pos -= 1

        if self.mask > 0:
            self.add_masking(x, label_ids)

        return x, span_labels, label_ids, label_counts

class CustomCollator():

    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def my_collate(self, labels, labels_length):
        for i, label in enumerate(labels):
            labels[i] = torch.nn.functional.pad(label, pad=(0,labels_length-len(labels[i])), value=-1)

        return torch.stack(labels)

    def __call__(self, batch):
        text = []
        span_labels = []
        label_ids = []
        label_counts = []

        for item in batch:
            text.append(item[0])
            span_labels.append(item[1])
            label_ids.append(item[2])
            label_counts.append(item[3])

        text = self.data_collator(text)
        text_length = text['input_ids'].size(dim=1)

        span_labels_length = max([len(l) for l in span_labels])

        span_labels = self.my_collate(span_labels, span_labels_length)
        label_ids = self.my_collate(label_ids, text_length)
        label_counts = self.my_collate(label_counts, span_labels_length)

        return text, span_labels, label_ids, label_counts

class PredictTextDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):

        self.tokenizer = tokenizer
        self.max_length = cfg['max_length']
        self.cfg = cfg

        self.texts = df['text'].values.tolist()
        self.data_labels = df['labels'].values.tolist()
        self.stride = cfg['stride']

        self.x, self.label_ids, self.bio_labels = [], [], []
        self.x_cut, self.label_ids_cut, self.bio_labels_cut = [], [], []
        self.text_indexes = []

        text_index = 0

        for idx in range(len(self.texts)):
            x, label_ids, bio_labels = self.make_item(idx)

            self.x.append(x)
            self.label_ids.append(label_ids)
            self.bio_labels.append(bio_labels)

            start = 0
            total_tokens = len(bio_labels)

            break_bool = False

            while start < total_tokens and not break_bool:

                if start + self.max_length > total_tokens:
                    start = max(0, total_tokens - self.max_length)
                    break_bool = True

                x_cut, label_ids_cut, bio_labels_cut = self.get_cut_item(x, label_ids, bio_labels, start)

                self.x_cut.append(x_cut)
                self.label_ids_cut.append(label_ids_cut)
                self.bio_labels_cut.append(bio_labels_cut)

                self.text_indexes.append((text_index, start))

                start += self.stride

            text_index += 1

    def get_cut_element(self, tokenized_element, start, length):
        return tokenized_element[start:start+length].clone()

    def get_cut_item(self, tokenized, label_ids, bio_labels, start):

        cut_length = min(self.max_length, len(bio_labels))

        new_tokenized = {}

        for k in tokenized:
            new_tokenized[k] = self.get_cut_element(tokenized[k], start, cut_length)

        if label_ids is not None:
            label_ids = self.get_cut_element(label_ids, start, cut_length)

        if bio_labels is not None:
            bio_labels = self.get_cut_element(bio_labels, start, cut_length)

        return new_tokenized, label_ids, bio_labels

    def __len__(self):
        return len(self.x_cut)

    def make_one_hot(self, bio_labels):
        one_hot_labels = torch.zeros((len(bio_labels), len(discourse_type_to_label)), dtype=torch.long)

        for idx, label in enumerate(bio_labels):
            if label != -1:
                one_hot_labels[idx][label] = 1

        return one_hot_labels

    def make_item(self, idx):

        text = self.texts[idx]

        tokenized = self.tokenizer(
            text,
            add_special_tokens = True,
            return_offsets_mapping=True,
            )

        offset_mapping = tokenized['offset_mapping']
        del tokenized['offset_mapping']

        skip_indices = np.where(np.array(tokenized.sequence_ids()) != 0)[0]
        bio_labels = np.zeros(len(offset_mapping), dtype=np.int)
        bio_labels[skip_indices] = -1

        label_ids = torch.full((len(offset_mapping),), -1, dtype=torch.long)

        for label_id, (start, end, discourse_type, _) in enumerate(self.data_labels[idx]):

            target_idx = []

            for token_idx, (off_start, off_end) in enumerate(offset_mapping):
                if min(end, off_end) > max(start, off_start):
                    label_ids[token_idx] = label_id
                    target_idx.append(token_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1] + 1

            pred_start = discourse_type_to_label['B-' + discourse_type]
            pred_end = discourse_type_to_label['I-' + discourse_type]

            bio_labels[targets_start : targets_end] = [pred_end] * (targets_end - targets_start)

            bio_labels[targets_start] = pred_start

        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v, dtype=torch.long)

        bio_labels = torch.tensor(bio_labels, dtype=torch.long)

        return tokenized, label_ids, bio_labels

    def __getitem__(self, idx):
        return self.x_cut[idx], self.label_ids_cut[idx], self.bio_labels_cut[idx], self.make_one_hot(self.bio_labels_cut[idx])

class PredictCustomCollator():

    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def my_collate(self, labels, labels_length):
        for i, label in enumerate(labels):
            labels[i] = torch.nn.functional.pad(label, pad=(0,labels_length-len(labels[i])), value=-1)

        return torch.stack(labels)

    def __call__(self, batch):
        text = []
        label_ids = []
        bio_labels = []
        one_hot_bio_labels = []

        for item in batch:
            text.append(item[0])
            label_ids.append(item[1])
            bio_labels.append(item[2])
            one_hot_bio_labels.append(item[3])

        text = self.data_collator(text)
        text_length = text['input_ids'].size(dim=1)

        label_ids = self.my_collate(label_ids, text_length)
        bio_labels = self.my_collate(bio_labels, text_length)

        for i, label in enumerate(one_hot_bio_labels):
            one_hot_bio_labels[i] = torch.nn.functional.pad(label, pad=(0, 0, 0,text_length-len(one_hot_bio_labels[i])), value=0)

        one_hot_bio_labels = torch.stack(one_hot_bio_labels)

        return text, label_ids, bio_labels, one_hot_bio_labels

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df = None,
        valid_df = None,
        predict_df = None,
        tokenizer = None,
        cfg = None,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.predict_df = predict_df

        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage):
        self.train_dataset = TextDataset(self.train_df, self.tokenizer, self.cfg, validation=False)
        self.valid_dataset = TextDataset(self.valid_df, self.tokenizer, self.cfg, validation=True)

    def train_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.train_dataset, **self.cfg["train_loader"], collate_fn=custom_collator)

    def val_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)

    def predict_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)
