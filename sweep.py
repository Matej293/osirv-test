import os
import wandb
import torch
import yaml
import argparse
from predict import train_model, evaluate_model
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from network import modeling
from metrics.wandb_logger import WandbLogger

def sweep_train(sweep_config=None):
    with wandb.init(config=sweep_config):
        wandb_params = wandb.config

        config_manager = ConfigManager("config/default_config.yaml")
        args = argparse.Namespace()
        
        # Override default config with sweep parameters
        # Map wandb sweep parameters to the format expected by update_from_args
        if 'training.learning_rate' in wandb_params:
            args.lr = float(wandb_params['training.learning_rate'])
        if 'training.weight_decay' in wandb_params:
            args.weight_decay = float(wandb_params['training.weight_decay'])
        if 'data.batch_size' in wandb_params:
            args.batch_size = int(wandb_params['data.batch_size'])
        if 'training.ssa_threshold' in wandb_params:
            args.ssa_threshold = float(wandb_params['training.ssa_threshold'])
        if 'training.hp_threshold' in wandb_params:
            args.hp_threshold = float(wandb_params['training.hp_threshold'])

        # manually set the epochs for sweep runs
        args.epochs = 10
        
        config_manager.update_from_args(args)
        
        aug_params = {}
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
                param_name = key.split('.')[-1]
                aug_params[param_name] = wandb_params[key]
                
                parts = key.split('.')
                current = config_manager.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = wandb_params[key]
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        train_loader = get_mhist_dataloader(
            config_manager.get('data.csv_path'),
            config_manager.get('data.img_dir'),
            config_manager.get('data.batch_size'),
            partition="train",
            augmentation_config=config_manager.get('augmentation.train')
        )
        
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
                    print(f"Loaded model from {model_path}")
                else:
                    print(f"Warning: Invalid checkpoint format in {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        model.to(device)
        
        # print the hyperparameters
        print("\nTraining with hyperparameters:")
        print(f"Learning Rate: {config_manager.get('training.learning_rate')}")
        print(f"Weight Decay: {config_manager.get('training.weight_decay')}")
        print(f"Epochs: {config_manager.get('training.epochs')}")
        print(f"Batch Size: {config_manager.get('data.batch_size')}")
        print(f"SSA Threshold: {config_manager.get('training.ssa_threshold')}")
        print(f"HP Threshold: {config_manager.get('training.hp_threshold')}")
        print("\nAugmentation parameters:")
        for key, value in config_manager.get('augmentation.train').items():
            print(f"{key}: {value}")
        
        logger = WandbLogger(
            project="mhist-classification",
            name=f"sweep-run-{wandb.run.id}",
            config=config_manager.config
        )
        
        train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            config=config_manager,
            logger=logger,
            save_path=f"models/sweep_{wandb.run.id}.pth"
        )

        final_epoch = config_manager.get('training.epochs') - 1

        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            config=config_manager,
            logger=logger,
            step=final_epoch
        )

if __name__ == "__main__":
    with open("config/sweep_config.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="mhist-classification")
    print(f"Sweep initialized with ID: {sweep_id}")
    
    wandb.agent(sweep_id, function=sweep_train, count=200)