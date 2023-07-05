model_path = 'checkpoints/240_1_PatchTST_custom_ftM_sl240_ll48_pl1_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'

import pandas as pd

df = pd.read_csv('../BTCUSDT_1h_features.csv')

from models import PatchTST
import torch
import argparse

configs = {
    'enc_in': 862,
    'seq_len': 240,
    'pred_len': 1,
    'e_layers': 3,
    'n_heads': 16,
    'd_model': 128,
    'd_ff': 256,
    'dropout': 0.2,
    'fc_dropout': 0.2,
    'head_dropout': 0,
    'individual': 0,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": 'end',
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25
}
configs_obj = argparse.Namespace(**configs)

model = PatchTST.Model(configs_obj).float()
# Load in the weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))

print(df.head())

