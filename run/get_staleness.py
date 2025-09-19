import pandas as pd
import matplotlib.pyplot as plt
import wandb
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run_path', type=str, required=True, help='WandB run path (e.g., user/project/run_id)')
args = parser.parse_args()
api = wandb.Api()
run = api.run(args.run_path)
df = pd.DataFrame(run.scan_history())
df['staleness'] = (df['train/iw_after_clip'].apply(lambda x: abs(np.log(x))))
print(f"Average Staleness: {df['staleness'].mean()}")