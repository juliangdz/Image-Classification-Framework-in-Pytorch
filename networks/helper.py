import pdb
import torch
from data.helper import get_sample_images
from tqdm import tqdm
import numpy as np 

def evaluate(model, data_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, step, tag='Eval'):
    model.eval()
    total_loss = 0
    all_metrics = {}
    all_preds, all_targets, all_images = [], [], []
    log_images = True

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"{tag}", leave=False)  # tqdm progress bar for evaluation
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            metrics = metrics_manager.compute_metrics(output, target,tag)
            for key, value in metrics.items():
                try:
                    all_metrics[key] += value
                except Exception:
                    all_metrics[key]=value

            preds = torch.argmax(output, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if log_images:
                all_images.extend(data.cpu())
                log_images = False

    if all_images:
        tensorboard_cb.log_evaluation_images(torch.stack(all_images), all_preds[0], all_targets[0], step, tag)
        wandb_cb.log_evaluation_images(torch.stack(all_images), all_preds[0], all_targets[0], tag, step)

    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    confusion_mat = metrics_manager.compute_confusion_matrix(np.concatenate(all_preds), np.concatenate(all_targets))
    return avg_metrics, confusion_mat


def train(network_name,model, tensorboard_cb, wandb_cb, train_loader, val_loader, test_loader, optimizer, criterion, metrics_manager, epochs, device, log_interval=10):
    sample_images, sample_labels = get_sample_images(train_loader)
    tensorboard_cb.log_images(sample_images, sample_labels, tag='Train Samples')
    wandb_cb.log_images(sample_images, sample_labels, tag='Train Samples')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)  # tqdm progress bar
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                step = epoch * len(train_loader) + batch_idx
                metrics = metrics_manager.compute_metrics(output, target,'Train')
                # Update progress bar with loss information
                progress_bar.set_postfix({'loss': loss.item()})
                tensorboard_cb.log_training(loss.item(), metrics, step)
                wandb_cb.log({"Loss/Train": loss.item(), **metrics}, step)
                tensorboard_cb.log_model_parameters(model, step)
                
                if network_name != 'fcn':
                    sample_images, _ = get_sample_images(train_loader)  
                    tensorboard_cb.log_feature_maps(model, sample_images.to(device), step)
        
        total_training_steps = (epoch + 1) * len(train_loader)
        val_metrics, val_confusion_matrix = evaluate(model, val_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, total_training_steps, tag='Val')
        tensorboard_cb.log_validation(val_metrics['loss'], val_metrics, total_training_steps)
        wandb_cb.log({"Loss/Val": val_metrics['loss'], **val_metrics}, total_training_steps)
        tensorboard_cb.log_confusion_matrix(val_confusion_matrix, total_training_steps)
        wandb_cb.log_confusion_matrix(val_confusion_matrix, total_training_steps)

    test_metrics, test_confusion_matrix = evaluate(model, test_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, total_training_steps, tag='Test')
    tensorboard_cb.log_test(test_metrics['loss'], test_metrics, total_training_steps)
    wandb_cb.log_test({"Loss/Test": test_metrics['loss'], **test_metrics}, total_training_steps)
    tensorboard_cb.log_confusion_matrix(test_confusion_matrix, total_training_steps)
    wandb_cb.log_confusion_matrix(test_confusion_matrix, total_training_steps)
    
    if network_name != 'fcn':
        # apply gradcam 
        test_images, test_labels = get_sample_images(test_loader)
        wandb_cb.apply_gradcam_and_log_batch(model,test_images,device,total_training_steps,tag='Test/GradCam') # comment out if using no conv layers
