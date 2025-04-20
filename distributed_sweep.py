import os
import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import argparse
from predict import train_model, evaluate_model
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from network import modeling
from metrics.wandb_logger import WandbLogger

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def distributed_sweep_train(rank, world_size, sweep_config=None, sweep_id=None):
    """Run training on a specific GPU."""
    setup(rank, world_size)
    
    with wandb.init(config=sweep_config) as run:
        wandb_params = wandb.config
        
        config_manager = ConfigManager("config/default_config.yaml")
        args = argparse.Namespace()
        
        # Override default config with sweep parameters
        if 'training.learning_rate' in wandb_params:
            args.lr = float(wandb_params['training.learning_rate'])
        if 'training.weight_decay' in wandb_params:
            args.weight_decay = float(wandb_params['training.weight_decay'])
        if 'data.batch_size' in wandb_params:
            # per-gpu batch size
            args.batch_size = int(wandb_params['data.batch_size']) // world_size
        if 'training.threshold' in wandb_params:
            args.threshold = float(wandb_params['training.threshold'])

        # manually set the epochs for sweep runs
        args.epochs = 15
        
        config_manager.update_from_args(args)
        
        aug_param_keys = [
            'augmentation.train.rotation_degrees',
            'augmentation.train.brightness',
            'augmentation.train.contrast',
            'augmentation.train.saturation',
            'augmentation.train.horizontal_flip_prob',
            'augmentation.train.vertical_flip_prob'
        ]
        
        for key in aug_param_keys:
            if key in wandb_params:
                parts = key.split('.')
                current = config_manager.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = wandb_params[key]
        
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        
        if rank == 0:
            print(f"Using {world_size} GPUs for training")
        
        train_dataset = get_mhist_dataloader(
            config_manager.get('data.csv_path'),
            config_manager.get('data.img_dir'), 
            config_manager.get('data.batch_size') // world_size,
            partition="train",
            augmentation_config=config_manager.get('augmentation.train'),
            distributed=True,
            world_size=world_size,
            rank=rank,
            return_dataset=True
        )
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config_manager.get('data.batch_size') // world_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Test loader doesn't need to be distributed
        test_loader = get_mhist_dataloader(
            config_manager.get('data.csv_path'),
            config_manager.get('data.img_dir'),
            config_manager.get('data.batch_size'),
            partition="test",
            augmentation_config=config_manager.get('augmentation.test')
        )
        
        model = modeling.deeplabv3plus_resnet101(
            num_classes=config_manager.get('model.num_classes'),
            output_stride=config_manager.get('model.output_stride'),
            pretrained_backbone=config_manager.get('model.pretrained_backbone')
        )
        
        model.classifier = modeling.DeepLabHeadV3Plus(
            in_channels=2048,
            low_level_channels=256,
            num_classes=config_manager.get('model.num_classes'),
            aspp_dilate=config_manager.get('model.aspp_dilate')
        )
        
        model_path = config_manager.get('model.pretrained_path')
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                    if 'classifier.classifier.3.weight' in state_dict:
                        del state_dict['classifier.classifier.3.weight']
                    if 'classifier.classifier.3.bias' in state_dict:
                        del state_dict['classifier.classifier.3.bias']
                    model.load_state_dict(state_dict, strict=False)
                    if rank == 0:
                        print(f"Loaded model from {model_path}")
                else:
                    if rank == 0:
                        print(f"Warning: Invalid checkpoint format in {model_path}")
            except Exception as e:
                if rank == 0:
                    print(f"Error loading model: {e}")
        
        model.to(device)
        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        
        # master process logging only
        if rank == 0:
            print("\nTraining with hyperparameters:")
            print(f"Learning Rate: {config_manager.get('training.learning_rate')}")
            print(f"Weight Decay: {config_manager.get('training.weight_decay')}")
            print(f"Epochs: {config_manager.get('training.epochs')}")
            print(f"Batch Size: {config_manager.get('data.batch_size')} (per process: {config_manager.get('data.batch_size') // world_size})")
            print(f"Threshold: {config_manager.get('training.threshold')}")
            print("\nAugmentation parameters:")
            for key, value in config_manager.get('augmentation.train').items():
                print(f"{key}: {value}")
            
            logger = WandbLogger(
                project="mhist-classification-2",
                name=f"sweep-run-{wandb.run.id}",
                config=config_manager.config
            )
        else:
            logger = None
        
        train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            config=config_manager,
            logger=logger if rank == 0 else None,
            save_path=f"models/sweep_{wandb.run.id}.pth" if rank == 0 else None,
            distributed=True,
            train_sampler=train_sampler
        )

        if rank == 0:
            final_epoch = config_manager.get('training.epochs')
            
            model_module = model.module
            model_module.load_state_dict(torch.load(f"models/sweep_{wandb.run.id}.pth"))

            evaluate_model(
                model=model_module,
                test_loader=test_loader,
                device=device,
                config=config_manager,
                logger=logger,
                step=final_epoch
            )

    cleanup()

def run_sweep():
    with open("config/sweep_config.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="mhist-classification-2")
    print(f"Sweep initialized with ID: {sweep_id}")

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs, training with distributed data parallel")
        
        def distributed_agent():
            mp.spawn(
                distributed_sweep_train,
                args=(world_size, sweep_config, sweep_id),
                nprocs=world_size,
                join=True
            )
        
        wandb.agent(sweep_id, function=distributed_agent, count=50)
        from sweep import cleanup_wandb_directories
        cleanup_wandb_directories()
        print("Sweep completed. All runs cleaned up.")
    else:
        print("Only one GPU found, using single GPU training")
        from sweep import sweep_train, cleanup_wandb_directories
        wandb.agent(sweep_id, function=sweep_train, count=50)
        cleanup_wandb_directories()
        print("Sweep completed. All runs cleaned up.")
        

if __name__ == "__main__":
    run_sweep()