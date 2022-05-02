# -*- coding: utf-8 -*-
"""
Reference:
Original file is located at: https://colab.research.google.com/github/woven-planet/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb

Code adapted and modified.
"""

from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from torchvision.models.inception import inception_v3,BasicConv2d
from torchvision.models import Inception3
from torchvision.transforms import Resize
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path
import subprocess
import os


rc = subprocess.call('/home/trdhasade/inception/setup_local_data.sh')
# # set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = '/home/trdhasade/ExtractedDataset'
print(os.environ["L5KIT_DATA_FOLDER"])

dm = LocalDataManager(None)
# get config
cfg = load_config_data("/home/trdhasade/inception/agent_motion_config_inception.yaml")
print(cfg)

"""## Model
"""
def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = inception_v3(pretrained=False)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.Conv2d_1a_3x3 = BasicConv2d(num_in_channels,32, kernel_size=5, stride=2, padding = 1)
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.AuxLogits.fc = nn.Linear(768, num_targets)
    model.fc = nn.Linear(2048, num_targets)

    return model

def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    print("Old",inputs.shape)
    transform = Resize((299,299))
    inputs=transform(inputs)
    print("New",inputs.shape)
    # outputs = model(inputs).reshape(targets.shape)
    print("output",model(inputs))
    outputs, aux_outputs = model(inputs)
    outputs = outputs.reshape(targets.shape)
    aux_outputs = aux_outputs.reshape(targets.shape)

    print("watch",outputs.shape,targets.shape)
    loss1 = criterion(outputs, targets)
    loss2 = criterion(aux_outputs, targets)
    loss = loss1 + 0.4*loss2
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs

def forward_eval(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    print("Old",inputs.shape)
    transform = Resize((299,299))
    inputs=transform(inputs)
    print("New",inputs.shape)
    # outputs = model(inputs).reshape(targets.shape)
    print("output",model(inputs))
    outputs = model(inputs)
    outputs = outputs.reshape(targets.shape)
    # aux_outputs = aux_outputs.reshape(targets.shape)

    print("watch",outputs.shape,targets.shape)
    loss1 = criterion(outputs, targets)
    # loss2 = criterion(aux_outputs, targets)
    loss = loss1     
    # loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs

"""## Load the Train Data
"""

# ===== INIT DATASET
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
print(train_dataset)

# ==== INIT MODEL
# torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="none")

"""# Training
"""

# ==== TRAIN LOOP
tr_it = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
losses_avg = []

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    loss, _ = forward(data, model, device, criterion)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    losses_avg.append(np.mean(losses_train))
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

"""### Save the Model and Loss 
We can plot the train loss against the iterations (batch-wise)
"""
model.eval()
torch.set_grad_enabled(False)

import pandas as pd
dict = {'losses':losses_train, 'losses_avg':losses_avg}
df=pd.DataFrame(dict)
df.to_csv('/home/trdhasade/inception/inception_loss.csv')


torch.save(model, '/home/trdhasade/Out/inception_fulltrain.pth')
