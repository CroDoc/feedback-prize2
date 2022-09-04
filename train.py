import argparse
import os
import random
from code.data import get_train
from code.dataset import TextDataModule
from code.model import TextModel
from code.score import Scorer

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.checkpoint
import yaml
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default=None, action='store', required=True
    )

    parser.add_argument(
        '--split', default=None, action='store', required=True
    )

    parser.add_argument(
        '--yaml', default=None, action='store', required=True
    )

    parser.add_argument(
        '--fold', default=None, type=int, required=False
    )

    parser.add_argument(
        '--device', default=1, type=int, required=False
    )

    parser.add_argument(
        '--loss', default='ce', action='store', required=False
    )

    parser.add_argument(
        '--head', default='linear', action='store', required=False
    )

    parser.add_argument(
        '--pretrained', default=None, action='store', required=False
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

with open(opt.yaml, 'r') as f:
    cfg = yaml.safe_load(f)

for key,value in vars(opt).items():

    if key in ['loss', 'head']:
        if key == 'loss':
            if value == 'ce':
                value = 'nn.CrossEntropyLoss'
            else:
                raise NotImplemented()
        cfg['model'][key] = value
    else:
        cfg[key] =  value

root_dir = 'runs/' + cfg['split'] + '/' + cfg['name']
try:
    os.makedirs(root_dir)
except:
    pass

df = get_train()

df_fold = pd.read_csv("data/df_folds.csv")
df = df.merge(df_fold,how='left',on="essay_id")

fold_name = 'fold_k_5_seed_42'

cfg['num_folds'] = max(df[fold_name]) + 1

yaml.dump(cfg, open(root_dir + '/hparams.yml', 'w'))

model_name = cfg['model']['model_name']

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(root_dir + '/tokenizer/')

for fold in range(cfg['num_folds']):

    if not opt.fold is None and opt.fold != fold:
        continue

    train_df = df[df[fold_name] != fold].reset_index(drop=True)
    valid_df = df[df[fold_name] == fold].reset_index(drop=True)

    cfg['dataset_size'] = len(train_df)

    datamodule = TextDataModule(train_df=train_df, valid_df=valid_df, tokenizer=tokenizer, cfg=cfg)

    model = TextModel(cfg, Scorer(datamodule))

    torch.save(model.config, root_dir + '/config.pth')

    if opt.pretrained:
        pl_fold = random.randint(0, 7)

        print('USING PRETRAINED FOLD:', pl_fold)

        state_dict = torch.load(opt.pretrained + '/fold' + str(pl_fold) + '.pth')
        del state_dict['fc.bias']
        del state_dict['fc.weight']
        del state_dict['fc_seg.bias']
        del state_dict['fc_seg.weight']

        for key in list(state_dict.keys()):
            state_dict[key.replace('model.deberta.', '')] = state_dict.pop(key)

        model.backbone.load_state_dict(state_dict, strict=True)

    earystopping = EarlyStopping(
        monitor = 'score',
        patience = cfg['callbacks']['patience'],
        mode = 'min',
    )

    callback_list = [earystopping]

    loss_weights = callbacks.ModelCheckpoint(
        dirpath=root_dir + '/weights',
        filename='fold=' + str(fold) + '-{epoch}-{score:.4f}' + '_weights',
        monitor='score',
        save_weights_only=True,
        save_top_k=1,
        mode='min',
        save_last=False,
    )
    callback_list.append(loss_weights)

    trainer = pl.Trainer(
        logger=None,
        max_epochs=cfg['epoch'],
        callbacks=callback_list,
        accelerator='gpu',
        devices=opt.device,
        **cfg['trainer'],
    )

    trainer.fit(model, datamodule=datamodule)
