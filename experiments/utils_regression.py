import subprocess
import datetime
import os

import pandas as pd
from absl import logging


def exec_bash(cmd: str):
  def gen_exec(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
      yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
      raise subprocess.CalledProcessError(return_code, cmd)

  for path in gen_exec(cmd.split()):
    print(path, end="")


def datetime_str(sep=""):
  str_format = f'{sep.join([f"%{t}" for t in "ymd"])}_{sep.join([f"%{t}" for t in "HMS"])}'
  return datetime.datetime.now().strftime(str_format)



def res_to_csv(flags_dict, total_results, hash=None):

  file_path = f"{flags_dict['output_dir']}/{flags_dict['results_file']}"
  if hash is not None:
    file_path = "".join(file_path.split(".csv")[:-1]+["_",hash, ".csv"])


  file_path = fix_path(file_path, False)

  logging.info(f"Saving results to {file_path}")

  res_dict = {k: [flags_dict[k]] for k in
              ["model", "seed", "dataset", "loss_fn", "ensemble_size", "train_epochs", "base_learning_rate",
               "mlp_hidden_dim", "divide_l2_loss", "random_init"]}
  for k, v in total_results.items():
    res_dict[k] = [total_results[k]]
  res_df = pd.DataFrame.from_dict(res_dict)

  if os.path.exists(file_path):
    res_df = pd.concat([pd.read_csv(file_path, index_col=0), res_df]).reset_index(drop=True)
  res_df.to_csv(file_path)



def fix_path(path:str, is_folder:bool):
  p = "/".join([s for s in path.split("/") if s != '']).replace("gs:", "gs:/")
  if is_folder:
    return p+"/"
  return p

