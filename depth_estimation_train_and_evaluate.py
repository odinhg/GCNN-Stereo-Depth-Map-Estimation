import torch
import torch.nn as nn
from torchinfo import summary
from os.path import isfile, join

from models import GUNetModel
from utils import create_dataloaders, Trainer, seed_random_generators, choose_model, Group, d2_r, d2_mh, d2_mv, d2_e
from config import configs, val_fraction, test_fraction, device

seed_random_generators()
data_path = "data"
figs_path = "figs"
checkpoints_path = "checkpoints"
val_fraction = 0.10                 # Fraction of data to use for validation data
test_fraction = 0.20                # Fraction of data to use for test data
num_workers = 8                     # Number of workers to use with dataloader.
device = "cuda:4"                   # Device for PyTorch to use. Can be "cpu" or "cuda:n".
config = {
                "name" : "GUNet", 
                "batch_size" : 2,
                "lr" : 1e-4,
                "epochs" : 50,
                "val_per_epoch" : 4,
                "checkpoint_file" : join(checkpoints_path, "gunet.pth"),
                "loss_plot_file" : join(figs_path, "gunet_loss_plot.png"),
                "earlystop_limit" : 20
            }


# Data loaders
train_dl, val_dl, test_dl = create_dataloaders(batch_size=config["batch_size"], val=val_fraction, test=test_fraction)

# Functions and Cayley table representing the symmetry group of a rectangle
functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)
model = GUNetModel(group)

# Print summary of layers and number of parameters
summary(model)

# Initialize trainer with loss function and optimizer
loss_function = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-2) 
trainer = Trainer(model, train_dl, val_dl, test_dl, config, loss_function, device)

# Train model if checkpoint is not found
if not isfile(config["checkpoint_file"]):
    trainer.train(optimizer)
    trainer.summarize_training()

# Evaluate model on test data
trainer.evaluate()
