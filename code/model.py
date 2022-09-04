from code.utils import discourse_type_to_label, preds_to_span_preds

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import (AutoConfig, AutoModel,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)


class TextModel(pl.LightningModule):

    def __init__(self, cfg, scorer=None, config_path=None):
        super().__init__()

        self.cfg = cfg
        self.scorer = scorer
        self.model_cfg = cfg['model']

        self.criterion = eval(self.model_cfg['loss'])()

        model_name = self.model_cfg['model_name']

        if config_path:
            self.config = torch.load(config_path)
        else:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        if self.model_cfg['pretrained']:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.dropout = nn.Dropout(self.model_cfg['dropout'])

        hidden_size = self.config.hidden_size# + len(discourse_type_to_label)
        self.fc = nn.Linear(hidden_size, 3)

        if 'skip_validation' in cfg:
            self.skip_validation = cfg['skip_validation']
        else:
            self.skip_validation = 0

    def forward(self, inputs):
        x = self.backbone(**inputs).last_hidden_state
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def on_validation_epoch_start(self):
        print('\n')

    def on_validation_epoch_end(self):
        if self.skip_validation > 0:
            self.skip_validation -= 1

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        if self.skip_validation > 0:
            return {}

        return self.shared_step(batch)

    def shared_step(self, batch):
        text, span_labels, label_ids, label_counts = batch

        output = self(text)

        preds, new_span_labels, loss_divs = preds_to_span_preds(output, label_ids, span_labels, label_counts)

        loss = self.criterion(preds, new_span_labels.cuda())

        preds = output.detach().cpu()

        return {'loss': loss, 'preds': preds}

    def predict_step(self, batch, batch_idx):

        text, span_labels, label_ids = batch

        output = self(text)

        return output.detach().cpu()

    def validation_epoch_end(self, outputs):

        if self.skip_validation > 0:
            self.log(f'val_loss', 1.0, on_epoch=True, prog_bar=True)
            self.log(f'score', 1.0 * self.skip_validation, on_epoch=True, prog_bar=True)
            return

        preds = []
        loss = 0
        for out in outputs:
            if out['loss'] is not None:
                loss += out['loss']
                preds.extend([x for x in out['preds']])

        loss /= len(outputs)

        metrics = self.scorer.score(preds)

        self.log(f'val_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'score', metrics, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        weight_decay = self.cfg['optimizer']['weight_decay']

        param_optimizer = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if 'backbone' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if 'backbone' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if not 'backbone' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
            },
            {
                'params': [p for n, p in param_optimizer if not 'backbone' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
            },
        ]

        optimizer = eval(self.cfg['optimizer']['name'])(
            optimizer_parameters, **self.cfg['optimizer']['params']
        )

        if 'scheduler' in self.cfg:

            scheduler_name = self.cfg['scheduler']['name']
            params = self.cfg['scheduler']['params']

            if scheduler_name in ['poly', 'cosine_restart', 'cosine']:

                epoch_steps = self.cfg['dataset_size']
                batch_size = self.cfg['train_loader']['batch_size']

                warmup_steps = self.cfg['scheduler']['warmup'] * epoch_steps // batch_size
                training_steps = params['epochs'] * epoch_steps // batch_size

                if scheduler_name == 'poly':
                    power = params['power']
                    lr_end = params['lr_end']

                    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)
                elif scheduler_name == 'cosine_restart':
                    cycles = params['cycles']
                    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, warmup_steps, training_steps, cycles)
                else:
                    raise NotImplemented('not implemented!')
            else:
                scheduler = eval(scheduler_name)(
                    optimizer, **params
                )

            lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': self.cfg['scheduler']['interval'],
                'frequency': 1,
            }

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

        return optimizer
