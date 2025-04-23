import os
import wandb
import torch
import yaml
from predict import train_model, evaluate_model
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from network import modeling
from metrics.wandb_logger import WandbLogger

def sweep_train(sweep_config=None):
    with wandb.init(config=sweep_config):
        config_manager = ConfigManager("config/default_config.yaml")
        wandb_params = wandb.config
        
        for param_name, param_value in wandb_params.items():
            if param_value is not None:
                config_manager._update_nested(config_manager.config, param_name, param_value)
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # dataloader initialization
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
        
        # model initialization
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
        
        # loading the pretrained model if available
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
        
        # Print hyperparameters (optional)
        print("\nTraining with hyperparameters:")
        print(f"Learning Rate: {config_manager.get('training.learning_rate')}")
        print(f"Weight Decay: {config_manager.get('training.weight_decay')}")
        print(f"Epochs: {config_manager.get('training.epochs')}")
        print(f"Batch Size: {config_manager.get('data.batch_size')}")
        print(f"Threshold: {config_manager.get('training.threshold')}")
        print("\nAugmentation parameters:")
        for key, value in config_manager.get('augmentation.train').items():
            print(f"{key}: {value}")
        
        # wandb logger initialization
        logger = WandbLogger(
            project="mhist-classification-5",
            name=f"sweep-run-{wandb.run.id}",
            config=config_manager.config,
            reuse=True
        )
        
        # Train and evaluate
        train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            config=config_manager,
            logger=logger,
            save_path=f"models/sweep_{wandb.run.id}.pth"
        )

        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            config=config_manager,
            logger=logger,
            step=config_manager.get('training.epochs')
        )

def cleanup_wandb_directories():
    """Clean up wandb directories after sweep completion."""
    import shutil
    import os
    
    wandb_dir = os.environ.get("WANDB_DIR")
    
    if not wandb_dir or not os.path.exists(wandb_dir):
        print("Could not locate wandb directory")
        return
        
    print(f"Cleaning wandb runs directory...")
    shutil.rmtree(wandb_dir)
    os.makedirs(wandb_dir)
    print(f"Successfully cleaned wandb runs directory")

if __name__ == "__main__":
    with open("config/sweep_config.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="mhist-classification-5")
    print(f"Sweep initialized with ID: {sweep_id}")
    
    wandb.agent(sweep_id, function=sweep_train, count=15)

    cleanup_wandb_directories()
    print("Sweep completed. All runs cleaned up.")
