import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

def visualize_segmentation_results(images, masks, predictions, probabilities, step=None, logger=None, max_samples=6):
    """Create comprehensive visualization of segmentation results."""
    if logger is None or images is None or len(images) == 0:
        return
    
    batch_size = min(len(images), max_samples)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
    
    images_display = images[:batch_size].clone()
    images_display = images_display * std + mean
    images_display = images_display.clamp(0, 1)
    
    create_basic_visualizations(images_display, masks, predictions, step, logger, batch_size)
    create_detailed_visualizations(images_display, masks, predictions, probabilities, step, logger, batch_size)
    create_error_analysis(images_display, masks, predictions, step, logger, batch_size)
    create_uncertainty_visualization(images_display, probabilities, step, logger, batch_size)
    
    plt.close('all')


def create_basic_visualizations(images, masks, predictions, step, logger, batch_size):
    """Log input images, ground-truth masks, and predicted masks."""
    # 1) Inputs
    logger.log_images("Viz/Input", images, step=step, max_images=batch_size)
    
    # 2) Color masks
    gt_colored   = []
    pred_colored = []
    
    for i in range(batch_size):
        mask   = masks[i][0].cpu().numpy()
        pred   = predictions[i][0].cpu().numpy()
        h, w   = mask.shape

        cm_gt = np.zeros((3, h, w), dtype=np.float32)
        cm_gt[0] = mask
        #cm_gt[2] = 1.0 - mask
        gt_colored.append(torch.from_numpy(cm_gt))

        cm_pr = np.zeros((3, h, w), dtype=np.float32)
        cm_pr[0] = pred
        #cm_pr[2] = 1.0 - pred
        pred_colored.append(torch.from_numpy(cm_pr))
    
    gt_colored_tensor   = torch.stack(gt_colored)
    pred_colored_tensor = torch.stack(pred_colored)
    
    logger.log_images("Viz/GroundTruth", gt_colored_tensor,   step=step, max_images=batch_size)
    logger.log_images("Viz/Prediction",  pred_colored_tensor, step=step, max_images=batch_size)


def create_detailed_visualizations(images, masks, predictions, probabilities, step, logger, batch_size):
    """Create detailed side-by-side visualizations."""
    try:
        figs = []
        for i in range(batch_size):
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 3, figure=fig)
            
            img = images[i].permute(1, 2, 0).cpu().numpy()
            mask = masks[i][0].cpu().numpy()
            pred = predictions[i][0].cpu().numpy()
            prob = probabilities[i][0].cpu().numpy()
            
            # Original image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Ground truth mask - blue HP, red SSA
            ax2 = fig.add_subplot(gs[0, 1])
            if mask.mean() > 0.5:  # SSA (white/1)
                gt_display = np.zeros((mask.shape[0], mask.shape[1], 3))
                gt_display[:, :, 0] = mask  # Red channel
                ax2.set_title(f"Ground Truth: SSA")
            else:  # HP (black/0)
                gt_display = np.zeros((mask.shape[0], mask.shape[1], 3))
                gt_display[:, :, 2] = 1 - mask  # Blue channel
                ax2.set_title(f"Ground Truth: HP")
            ax2.imshow(gt_display)
            ax2.axis('off')
            
            # Prediction mask - blue HP, red SSA
            ax3 = fig.add_subplot(gs[0, 2])
            if pred.mean() > 0.5:
                pred_display = np.zeros((pred.shape[0], pred.shape[1], 3))
                pred_display[:, :, 0] = pred  # Red channel
                ax3.set_title(f"Prediction: SSA")
            else:
                pred_display = np.zeros((pred.shape[0], pred.shape[1], 3))
                pred_display[:, :, 2] = 1 - pred  # Blue channel 
                ax3.set_title(f"Prediction: HP")
            ax3.imshow(pred_display)
            ax3.axis('off')
            
            # Probability heatmap
            ax4 = fig.add_subplot(gs[1, 0])
            im = ax4.imshow(prob, cmap='jet', vmin=0, vmax=1)
            ax4.set_title("Probability Map (Red = SSA, Blue = HP)")
            ax4.axis('off')
            fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            
            # Overlay ground truth on image - blue HP, red SSA
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(img)
            overlay = np.zeros_like(img)
            if mask.mean() > 0.5:
                overlay[:, :, 0] = mask * 0.7  # Red for SSA
            else:
                overlay[:, :, 2] = (1 - mask) * 0.7  # Blue for HP
            ax5.imshow(overlay, alpha=0.5)
            ax5.set_title("Ground Truth Overlay")
            ax5.axis('off')
            
            # Overlay prediction on image - blue HP, red SSA
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(img)
            overlay = np.zeros_like(img)
            if pred.mean() > 0.5:
                overlay[:, :, 0] = pred * 0.7  # Red for SSA
            else:
                overlay[:, :, 2] = (1 - pred) * 0.7  # Blue for HP
            ax6.imshow(overlay, alpha=0.5)
            ax6.set_title("Prediction Overlay")
            ax6.axis('off')
            
            plt.tight_layout()
            figs.append(fig)
        
        # Log the figures
        for i, fig in enumerate(figs):
            logger._log_figure(f"Viz/DetailedSegmentation_{i+1}", fig, step=step)
            plt.close(fig)
            
    except Exception as e:
        plt.close('all')
        print(f"Warning: Could not create detailed segmentation visualization: {e}")

def create_error_analysis(images, masks, predictions, step, logger, batch_size):
    """Create error analysis visualizations."""
    try:
        for i in range(batch_size):
            mask_np = masks[i][0].cpu().numpy().astype(bool)
            pred_np = predictions[i][0].cpu().numpy().astype(bool)
            img = images[i].permute(1, 2, 0).cpu().numpy()
            
            # Create error map
            error_map = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
            error_map[np.logical_and(pred_np, mask_np)] = [1, 1, 1]  # True Positive: white
            error_map[np.logical_and(pred_np, np.logical_not(mask_np))] = [1, 0, 0]  # False Positive: red
            error_map[np.logical_and(np.logical_not(pred_np), mask_np)] = [0, 0, 1]  # False Negative: blue
            error_map[np.logical_and(np.logical_not(pred_np), np.logical_not(mask_np))] = [0, 0, 0]  # True Negative: black
            
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 1])
            
            # Original image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Error map
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(error_map)
            ax2.set_title("Error Analysis")
            ax2.axis('off')
            
            # Legend
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis('off')
            
            p1 = mpatches.Patch(color='white', label='True Positive: Correctly classified as SSA')
            p2 = mpatches.Patch(color='red', label='False Positive: Incorrectly classified as SSA (actually HP)')
            p3 = mpatches.Patch(color='blue', label='False Negative: Incorrectly classified as HP (actually SSA)')
            p4 = mpatches.Patch(color='black', label='True Negative: Correctly classified as HP')
            
            ax3.legend(handles=[p1, p2, p3, p4], loc='center', fontsize=12, frameon=True, 
                      title="Error Analysis Explanation", title_fontsize=14)
            
            plt.tight_layout()
            logger._log_figure(f"Viz/ErrorAnalysisDetailed_{i+1}", fig, step=step)
            plt.close(fig)
        
        # grid creation
        error_maps = []
        for i in range(batch_size):
            mask_np = masks[i][0].cpu().numpy().astype(bool)
            pred_np = predictions[i][0].cpu().numpy().astype(bool)
            
            error_map = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
            error_map[np.logical_and(pred_np, mask_np)] = [1, 1, 1]
            error_map[np.logical_and(pred_np, np.logical_not(mask_np))] = [1, 0, 0]
            error_map[np.logical_and(np.logical_not(pred_np), mask_np)] = [0, 0, 1]
            error_map[np.logical_and(np.logical_not(pred_np), np.logical_not(mask_np))] = [0, 0, 0]
            
            error_map = torch.from_numpy(error_map.transpose(2, 0, 1))
            error_maps.append(error_map)
        
        error_maps = torch.stack(error_maps)
        logger.log_images("Viz/ErrorAnalysis", error_maps, step=step, max_images=batch_size)
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Could not create error analysis visualization: {e}")

def create_uncertainty_visualization(images, probabilities, step, logger, batch_size):
    """Create uncertainty visualizations."""
    try:
        uncertainty_maps = []
        
        for i in range(batch_size):
            prob_np = probabilities[i][0].cpu().numpy()
            # uncertainty highest at = 0.5
            uncertainty = 1.0 - 2.0 * np.abs(prob_np - 0.5)
            
            cmap = plt.cm.get_cmap('viridis')
            colored_uncertainty = cmap(uncertainty)[:, :, :3]
            
            uncertainty_map = torch.from_numpy(colored_uncertainty.transpose(2, 0, 1))
            uncertainty_maps.append(uncertainty_map)
        
        uncertainty_maps = torch.stack(uncertainty_maps)
        logger.log_images("Viz/UncertaintyMap", uncertainty_maps, step=step, max_images=batch_size)
        
        # explanation figure
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[3, 1])
        
        # show 3 uncertainty maps
        for i in range(min(3, batch_size)):
            ax = fig.add_subplot(gs[0, i])
            prob_np = probabilities[i][0].cpu().numpy()
            uncertainty = 1.0 - 2.0 * np.abs(prob_np - 0.5)
            im = ax.imshow(uncertainty, cmap='viridis')
            ax.set_title(f"Uncertainty Map {i+1}")
            ax.axis('off')
            
        cbar_ax = fig.add_axes([0.92, 0.5, 0.02, 0.35])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Uncertainty Level')
        
        # explanation
        ax_exp = fig.add_subplot(gs[1, :])
        ax_exp.axis('off')
        explanation_text = (
            "The uncertainty map shows how confident the model is in its prediction.\n\n"
            "Yellow/Green areas: High uncertainty (probability close to 0.5)\n"
            "Dark purple/blue areas: Low uncertainty (probability close to 0 or 1)\n\n"
            "For binary classification between HP and SSA, high uncertainty regions\n"
            "represent areas where the model struggles to determine the correct class."
        )
        ax_exp.text(0.5, 0.5, explanation_text, ha='center', va='center', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9))
        
        logger._log_figure("Viz/UncertaintyExplanation", fig, step=step)
        plt.close(fig)
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Could not create uncertainty visualization: {e}")