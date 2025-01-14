import torch
from data.helper import get_sample_images,get_balanced_sample
from tqdm import tqdm
import numpy as np
import os
from utils.export_onnx import export_model_to_onnx

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after
    a given patience.
    """
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def evaluate(model, data_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, step, tag='Eval'):
    model.eval()
    total_loss = 0
    all_images = []
    log_images = True
    
    # Instead of computing metrics per batch, just accumulate them
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"{tag}", leave=False)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)  # shape: [batch_size, num_classes]
            loss = criterion(output, target)
            total_loss += loss.item()

            # Accumulate outputs for full-dataset metric computation
            all_outputs.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

            if log_images:
                # log just a small set of images
                all_images.extend(data.cpu())
                log_images = False

    # Convert all_outputs/all_targets to full-dataset arrays or tensors
    all_outputs_np = np.concatenate(all_outputs, axis=0)  # shape: [num_samples, num_classes]
    all_targets_np = np.concatenate(all_targets, axis=0)  # shape: [num_samples]

    # For confusion matrix, we do need discrete predictions:
    all_preds = np.argmax(all_outputs_np, axis=1)

    if len(all_images) > 0:
        tensorboard_cb.log_evaluation_images(
            torch.stack(all_images), 
            all_preds[: len(all_images)],  # just first batch’s worth
            all_targets_np[: len(all_images)],  
            step, 
            tag
        )
        wandb_cb.log_evaluation_images(
            torch.stack(all_images), 
            all_preds[: len(all_images)], 
            all_targets_np[: len(all_images)], 
            tag, 
            step
        )

    # Now we compute average loss across all batches
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches

    # Convert them to torch tensors for your metrics_manager
    outputs_tensor = torch.from_numpy(all_outputs_np).float()
    targets_tensor = torch.from_numpy(all_targets_np).long()
    
    # Compute full-dataset metrics
    avg_metrics = metrics_manager.compute_metrics(outputs_tensor, targets_tensor, tag)
    cls_report = metrics_manager.clf_report(outputs_tensor, targets_tensor)
    avg_metrics['loss'] = avg_loss

    # Confusion matrix
    confusion_mat = metrics_manager.compute_confusion_matrix(all_preds, all_targets_np)
    
    # Then log it to W&B
    wandb_cb.log_classification_report(cls_report, step,tag)

    return avg_metrics, confusion_mat

def train(network_name, model, tensorboard_cb, wandb_cb, train_loader, val_loader, test_loader, class_names,
          optimizer, criterion, metrics_manager, epochs, device, log_interval=10,
          checkpoint_dir="checkpoints", early_stopping_patience=5):
    """
    The main training loop with optional early stopping and checkpointing.
    """
    # Set up early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Log some sample images
    tran_loader_iter = iter(train_loader)
    sample_images, sample_labels = get_balanced_sample(tran_loader_iter,n_pos=32,n_neg=32)
    print(f"Input Shape: {sample_images.shape} - {sample_labels.shape}")
    tensorboard_cb.log_images(sample_images, sample_labels, tag='Train Samples')
    wandb_cb.log_images(sample_images, sample_labels, tag='Train Samples')

    total_training_steps = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                            desc=f"Epoch {epoch+1}/{epochs}", leave=False)
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
                progress_bar.set_postfix({'loss': loss.item()})
                tensorboard_cb.log_training(loss.item(), metrics, step)
                wandb_cb.log({"Loss/Train": loss.item(), **metrics}, step)
                tensorboard_cb.log_model_parameters(model, step)

                if network_name != 'fcn':
                    train_loader_iter = iter(train_loader)
                    sample_images, _ = get_balanced_sample(train_loader_iter,n_pos=32,n_neg=32)
                    tensorboard_cb.log_feature_maps(model, sample_images.to(device), step)

        total_training_steps = (epoch + 1) * len(train_loader)

        # Evaluate on validation set
        val_metrics, val_confusion_matrix = evaluate(model, val_loader, criterion, 
                                                     metrics_manager, device, 
                                                     tensorboard_cb, wandb_cb, 
                                                     total_training_steps, tag='Val')
        tensorboard_cb.log_validation(val_metrics['loss'], val_metrics, total_training_steps)
        wandb_cb.log({"Loss/Val": val_metrics['loss'], **val_metrics}, total_training_steps)
        tensorboard_cb.log_confusion_matrix(val_confusion_matrix, total_training_steps)
        wandb_cb.log_confusion_matrix(val_confusion_matrix, total_training_steps)

        # Checkpoint: Save best model
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss={val_loss:.4f}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Always save the last model (whether early stopped or not)
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved after epoch {epoch+1}")

    # Evaluate on test set after training
    test_metrics, test_confusion_matrix = evaluate(model, test_loader, criterion, 
                                                   metrics_manager, device, 
                                                   tensorboard_cb, wandb_cb, 
                                                   total_training_steps, tag='Test')
    tensorboard_cb.log_test(test_metrics['loss'], test_metrics, total_training_steps)
    wandb_cb.log_test({"Loss/Test": test_metrics['loss'], **test_metrics}, total_training_steps)
    tensorboard_cb.log_confusion_matrix(test_confusion_matrix, total_training_steps)
    wandb_cb.log_confusion_matrix(test_confusion_matrix, total_training_steps)

    # If network has conv layers
    if network_name != 'fcn':
        # GradCAM example on test images
        test_loader_iter = iter(test_loader)
        test_images, test_labels = get_balanced_sample(test_loader_iter,n_pos=32,n_neg=32)
        all_preds_for_test = model(test_images.to(device))  # shape: [B, num_classes]
        preds = all_preds_for_test.argmax(dim=1).cpu()
        targets = test_labels
        wandb_cb.apply_gradcam_and_log_batch(
            model,
            test_images,
            device,
            step=total_training_steps,
            tag='Test/GradCam',
            preds=preds,
            targets=targets,
            class_names=class_names
        )
        # wandb_cb.apply_gradcam_and_log_batch(model, test_images, device, total_training_steps, tag='Test/GradCam')
