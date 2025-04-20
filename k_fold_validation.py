import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm

from predict import train_model, evaluate_model
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from network import modeling
from metrics.wandb_logger import WandbLogger
from metrics.tensorboard_logger import TensorboardLogger


def k_fold_cross_validation(config_path="config/default_config.yaml", k=5, use_wandb=False, 
                            project_name="mhist-k-fold", random_state=42):
    config = ConfigManager(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    annotations_path = config.get('data.csv_path')
    df = pd.read_csv(annotations_path)
    all_images = df['Image Name'].tolist()
    
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    results = {
        "fold_metrics": [],
        "precision_values": [],
        "recall_values": [],
        "thresholds": [],
        "ap_scores": [],
        "auc_pr_scores": [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n{'='*50}\nFold {fold+1}/{k}\n{'='*50}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_csv_path = f"./temp_train_fold_{fold}.csv"
        val_csv_path = f"./temp_val_fold_{fold}.csv"
        
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
        
        if use_wandb:
            logger = WandbLogger(
                project=project_name,
                name=f"fold-{fold+1}",
                config=config.config
            )
            print(f"Using Weights & Biases for logging")
        else:
            log_dir = os.path.join(config.get('logging.log_dir'), f"fold_{fold+1}")
            logger = TensorboardLogger(log_dir=log_dir)
            print(f"Using TensorBoard for logging")
            
        train_loader = get_mhist_dataloader(
            train_csv_path,
            config.get('data.img_dir'),
            config.get('data.batch_size'),
            partition="train",
            augmentation_config=config.get('augmentation.train')
        )
        
        val_loader = get_mhist_dataloader(
            val_csv_path,
            config.get('data.img_dir'),
            config.get('data.batch_size'),
            partition="test",
            augmentation_config=config.get('augmentation.test')
        )
        
        model = modeling.deeplabv3plus_resnet101(
            num_classes=config.get('model.num_classes'),
            output_stride=config.get('model.output_stride'),
            pretrained_backbone=config.get('model.pretrained_backbone')
        )
        
        model.classifier = modeling.DeepLabHeadV3Plus(
            in_channels=2048,
            low_level_channels=256,
            num_classes=config.get('model.num_classes'),
            aspp_dilate=config.get('model.aspp_dilate')
        )
        
        model_path = config.get('model.pretrained_path')
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
            except Exception as e:
                print(f"Error loading model: {e}")
        
        model.to(device)
        
        save_path = f"./models/fold_{fold+1}_model.pth"
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            config=config,
            logger=logger,
            save_path=save_path
        )
        
        all_probs = []
        all_preds = []
        all_targets = []
        
        trained_model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Computing precision-recall data"):
                images, masks = images.to(device), masks.to(device)
                masks = masks.unsqueeze(1).float()
                
                outputs = trained_model(images)
                probs = torch.sigmoid(outputs)
                
                ssa_threshold = config.get('training.ssa_threshold')
                hp_threshold = config.get('training.hp_threshold')
                
                predicted = torch.zeros_like(masks)
                predicted[probs > ssa_threshold] = 1.0
                predicted[probs < hp_threshold] = 0.0
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
        ap_score = average_precision_score(all_targets, all_probs)
        auc_pr = auc(recall, precision)
        
        results["precision_values"].append(precision)
        results["recall_values"].append(recall)
        results["thresholds"].append(thresholds)
        results["ap_scores"].append(ap_score)
        results["auc_pr_scores"].append(auc_pr)
        
        print("\nPerforming final evaluation on validation fold...")
        dice, iou = evaluate_model(
            model=trained_model,
            test_loader=val_loader,
            device=device,
            config=config,
            logger=logger,
            step=config.get('training.epochs')
        )
        
        fold_results = {
            "fold": fold + 1,
            "dice": dice,
            "iou": iou,
            "ap_score": ap_score,
            "auc_pr": auc_pr
        }
        results["fold_metrics"].append(fold_results)
        
        for key, value in fold_results.items():
            if isinstance(value, (int, float)) and key != "fold":
                logger.log_scalar(f"KFold/{key}", value, step=fold+1)
        
        fig = plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, lw=2, label=f'Precision-Recall Curve (AP={ap_score:.3f}, AUC={auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Fold {fold+1} - Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        if hasattr(logger, '_log_figure'):
            logger._log_figure("KFold/PR_Curve", fig, step=fold+1)
        
        pr_curve_path = f"./pr_curve_fold_{fold+1}.png"
        plt.savefig(pr_curve_path)
        plt.close(fig)
        
        os.remove(train_csv_path)
        os.remove(val_csv_path)
        
        logger.close()
    
    avg_results = {}
    for metric in ["dice", "iou", "ap_score", "auc_pr"]:
        avg_results[f"avg_{metric}"] = np.mean([fold[metric] for fold in results["fold_metrics"]])
        avg_results[f"std_{metric}"] = np.std([fold[metric] for fold in results["fold_metrics"]])
    
    results["average_metrics"] = avg_results
    
    plt.figure(figsize=(10, 8))
    
    for fold in range(k):
        plt.plot(results["recall_values"][fold], results["precision_values"][fold], 
                 alpha=0.3, lw=1, label=f'Fold {fold+1}')
    
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    
    for fold in range(k):
        precision = results["precision_values"][fold]
        recall = results["recall_values"][fold]
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    
    plt.plot(mean_recall, mean_precision, color='b', lw=2, 
             label=f'Mean PR (AP={avg_results["avg_ap_score"]:.3f}±{avg_results["std_ap_score"]:.3f})')
    plt.fill_between(mean_recall, mean_precision - std_precision, 
                    mean_precision + std_precision, alpha=0.2, color='b', 
                    label=f'±1 std. dev.')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{k}-Fold Cross-Validation: Average Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    avg_pr_curve_path = "./average_pr_curve.png"
    plt.savefig(avg_pr_curve_path)
    plt.close()
    
    print("\n" + "="*50)
    print(f"K-Fold Cross-Validation Results (k={k})")
    print("="*50)
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)
    print(f"Average PR curve saved to {avg_pr_curve_path}")
    
    return results


def find_optimal_thresholds(results, f_beta=1.0):
    optimal_results = []
    
    for fold in range(len(results["precision_values"])):
        precision = results["precision_values"][fold]
        recall = results["recall_values"][fold]
        thresholds = results["thresholds"][fold]
        
        f_scores = ((1 + f_beta**2) * precision * recall) / (f_beta**2 * precision + recall + 1e-10)
        
        best_idx = np.argmax(f_scores)
        best_threshold = thresholds[best_idx]
        
        optimal_results.append({
            "fold": fold + 1,
            "optimal_threshold": best_threshold,
            "precision": precision[best_idx],
            "recall": recall[best_idx],
            "f_score": f_scores[best_idx]
        })
    
    avg_threshold = np.mean([r["optimal_threshold"] for r in optimal_results])
    std_threshold = np.std([r["optimal_threshold"] for r in optimal_results])
    
    return {
        "per_fold": optimal_results,
        "average_threshold": avg_threshold,
        "std_threshold": std_threshold,
        "f_beta": f_beta
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation with PR Curve Analysis')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--project', type=str, default='mhist-k-fold',
                        help='WandB project name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--f_beta', type=float, default=1.0,
                        help='Beta parameter for F-score calculation')
    
    args = parser.parse_args()
    
    results = k_fold_cross_validation(
        config_path=args.config,
        k=args.k,
        use_wandb=args.use_wandb,
        project_name=args.project,
        random_state=args.seed
    )
    
    threshold_results = find_optimal_thresholds(results, f_beta=args.f_beta)
    
    print("\n" + "="*50)
    print(f"Optimal Threshold Analysis (F{args.f_beta})")
    print("="*50)
    print(f"Average optimal threshold: {threshold_results['average_threshold']:.4f} ± {threshold_results['std_threshold']:.4f}")