# train.py
import pytorch_lightning as pl
from model import LitModel
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import datetime
import os
import wandb
import argparse
import sys
import yaml
import torch
import ast





base_dir = os.path.dirname(os.path.abspath(__file__))
interference_mapping = {
    'EMISignal1': 1,
    'CommSignal2': 2,
    'CommSignal3': 3,
    'CommSignal5G1': 5
}


local_rank = int(os.environ.get('LOCAL_RANK', 0))
# Initialize Weights & Biases only if this is the master process (i.e., local_rank == 0)
def train_model(timestamp):
        
    config_save_dir = f'/lightning_logs/{timestamp}'
    if local_rank == 0:
        os.makedirs(config_save_dir, exist_ok=True)
        config_file_path = os.path.join(config_save_dir, 'config.yaml')
        # Save the YAML config
        with open(config_file_path, 'w') as file:
            yaml.dump(OmegaConf.to_container(YAML_config, resolve=True), file)

    
    # YAML_config['dataloader']['interference_idx'] = interference_mapping.get(YAML_config['dataloader']['interference_type'])
    YAML_config['disable_sys_metrics'] = True
    YAML_config['timestamp'] = timestamp
    
    if YAML_config['trainer']['seed_everything']:
        pl.seed_everything(YAML_config.dataloader.seed0)
    f"Bit_Decoder"
    initialize_wandb(YAML_config, timestamp)
    # Max_Steps=YAML_config['trainer']['max_examples']//YAML_config['dataloader']['batch_size']

    
    
    if torch.cuda.device_count()>1 and YAML_config['distributed']['strategy']=="fsdp":
        YAML_config['distributed']['strategy']="ddp"
    
    n_devices = YAML_config['distributed']['n_devices']
    
    # Determine the number of GPUs
    print(n_devices)
    if n_devices == -1:
        num_gpus = torch.cuda.device_count()  # All available GPUs
    elif isinstance(n_devices, int):
        num_gpus = n_devices  # Specified number of GPUs
    else:
        num_gpus = len(n_devices)  # Number of GPUs in the list

        
    # Calculate the maximum number of epochs
    batch_size = YAML_config['dataloader']['batch_size']
    dataset_length = 734  # Replace with your actual dataset length
    max_steps = YAML_config['trainer']['max_steps']
    
    # Calculate steps per epoch considering the effective batch size in distributed training
    steps_per_epoch = dataset_length // (batch_size * num_gpus)
    
    # Calculate the maximum number of epochs
    max_epochs = max_steps // steps_per_epoch
    YAML_config['trainer']['max_epochs'] = max_epochs
    print(f'max_epochs={max_epochs} and max_steps={max_steps}')

    
    lit_model = LitModel(YAML_config)

    
#     checkpoint_callback = ModelCheckpoint(
#     dirpath=YAML_config['trainer']['model_save_dir'],
#     filename=YAML_config['timestamp'] + '-latest',
#     save_top_k=1,  # Save only the latest checkpoint
#     save_last=True,  # Always save the latest epoch's checkpoint
#     save_weights_only=True,  # Save only the model weights
#     every_n_epochs=YAML_config['trainer']['N_save_model']
# )
  #precision = bf16-mixed


    trainer = Trainer(
        accelerator='auto',  # this will automatically choose between CPU and GPU(s)   
        devices=YAML_config['distributed']['n_devices'],  # use all available GPUs
        strategy=YAML_config['distributed']['strategy'],  # using Distributed Data Parallel
        max_steps=YAML_config['trainer']['max_steps'],
        max_epochs=max_epochs,
        check_val_every_n_epoch=YAML_config['trainer']['val_check_interval'],
        # check_val_every_n_epoch=None,
        enable_checkpointing=False
        

    )

    
    trainer.fit(lit_model)





def fully_flatten_config(config):
    items = {}
    for k, v in config.items():
        if isinstance(v, dict):
            items.update(fully_flatten_config(v))
        else:
            items[k] = v
    return items


def initialize_wandb(YAML_config, timestamp):
    if local_rank == 0:
        wandb.login(key="your_wandb_key")
        # Start a new W&B run
        dict_config = fully_flatten_config(OmegaConf.to_container(YAML_config, resolve=True))
        wandb.init(project=YAML_config['wandb']['proj_name'], name=timestamp, config=dict_config)
    
        # Log any file (optional)
        wandb.save(os.path.join(base_dir, 'model.py'))
        wandb.save(os.path.join(base_dir, 'train.py'))
        wandb.save(os.path.join(base_dir, 'my_custom_loss.py'))
        wandb.save(os.path.join(base_dir, 'test_utils.py'))
        wandb.save(os.path.join(base_dir, 'test_online.py'))
        wandb.save(os.path.join(base_dir, 'data_module.py'))
        wandb.save(os.path.join(base_dir, 'TEST_data_module.py'))
        wandb.save(os.path.join(base_dir, 'other_models.py'))
        wandb.save(os.path.join(base_dir, 'model_wavenet.py'))



def parse_dynamic_args():
    parser = argparse.ArgumentParser(description="Dynamic Args Parser")
    parser.add_argument('--reference_config', type=str, help='Path to reference configuration file', default=None)
    parser.add_argument('args', nargs='*')
    parsed_args = parser.parse_args()

    args_dict = {}
    for arg in parsed_args.args:
        if '=' not in arg:
            print(f"Argument {arg} ignored. No '=' found.")
            continue
        key, value = arg.split('=', 1)

        try:
            # Use literal_eval to safely evaluate strings, lists, etc.
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if conversion fails
            pass

        args_dict[key] = value

    # Include the reference_config in the dictionary if provided
    if parsed_args.reference_config:
        args_dict['reference_config'] = parsed_args.reference_config    
    return args_dict




def update_nested_config(config, keys, value):
    for i, key in enumerate(keys):
        if i == len(keys) - 1:  # Last key, update the value
            if key in config or 1:
                config[key] = value
                return
            else:
                raise ValueError(f"Key '{'.'.join(keys)}' not found in the original config.")
        else:  # Intermediate key, navigate deeper
            if key in config or 1:
                config = config[key]
            else:
                raise ValueError(f"Intermediate key '{key}' not found in the config.")


def update_config(config, args_dict):
    for key, value in args_dict.items():
        nested_keys = key.split('.')
        update_nested_config(config, nested_keys, value)

if __name__ == "__main__":
    args_dict = parse_dynamic_args()
    # Determine the config file path
    if 'reference_config' in args_dict and args_dict['reference_config']:
        ref_timestamp=args_dict['reference_config']
        config_path = f'/lightning_logs/{ref_timestamp}/config.yaml'
        del args_dict['reference_config']
    else:
        config_path = os.path.join(base_dir, 'config.yaml')

    # Load the YAML configuration
    YAML_config = OmegaConf.load(config_path)    
    #YAML_config = OmegaConf.load(os.path.join(base_dir, 'config.yaml'))
    update_config(YAML_config, args_dict)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    conf_save_dir_stamped=f'/lightning_logs/{timestamp}'
    if local_rank == 0:
        os.makedirs(conf_save_dir_stamped)    
    train_model(timestamp)
