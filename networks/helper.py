from logger.log_tensorboard import TensorBoardCallback
from logger.log_wandb import WandBCallback
import torch
from data.helper import get_sample_images

def evaluate(model, data_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, step, tag='Eval'):
    model.eval()
    total_loss = 0
    all_metrics = {metric: 0 for metric in metrics_manager.metrics}
    all_preds, all_targets, all_images = [], [], []
    log_images = True  # Flag to log images only once

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            metrics = metrics_manager.compute_metrics(output, target)
            for key, value in metrics.items():
                all_metrics[key] += value

            preds = torch.argmax(output, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if log_images:
                all_images.extend(data.cpu())
                log_images = False  # Set to False after first batch

    # Log evaluation images with predictions and true labels
    if all_images:
        tensorboard_cb.log_evaluation_images(torch.stack(all_images), all_preds[0], all_targets[0], step, tag)
        wandb_cb.log_evaluation_images(torch.stack(all_images), all_preds[0], all_targets[0], tag, step)

    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    # Compute confusion matrix
    confusion_mat = metrics_manager.compute_confusion_matrix(torch.cat(all_preds), torch.cat(all_targets))
    return avg_metrics, confusion_mat


def train(model, train_loader, val_loader, test_loader, optimizer, criterion, metrics_manager, epochs, device, config, log_interval=10):
    tensorboard_cb = TensorBoardCallback(log_dir="./logs")
    wandb_cb = WandBCallback(config)

    # Log initial sample images from training set
    sample_images, sample_labels = get_sample_images(train_loader)
    tensorboard_cb.log_images(sample_images, sample_labels, tag='Train Samples')
    wandb_cb.log_images(sample_images, sample_labels, tag='Train Samples')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                step = epoch * len(train_loader) + batch_idx
                metrics = metrics_manager.compute_metrics(output, target)
                tensorboard_cb.log_training(loss.item(), metrics, step)
                wandb_cb.log({"Loss/Train": loss.item(), **metrics}, step)

        # Evaluate on validation data
        val_metrics, val_confusion_matrix = evaluate(model, val_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, epoch, tag='Validation')
        tensorboard_cb.log_validation(val_metrics['loss'], val_metrics, epoch)
        wandb_cb.log({"Loss/Val": val_metrics['loss'], **val_metrics}, epoch)
        tensorboard_cb.log_confusion_matrix(val_confusion_matrix, epoch)
        wandb_cb.log_confusion_matrix(val_confusion_matrix, epoch)

    # Evaluate on test data after all epochs
    test_metrics, test_confusion_matrix = evaluate(model, test_loader, criterion, metrics_manager, device, tensorboard_cb, wandb_cb, 'final_test', tag='Test')
    tensorboard_cb.log_test(test_metrics['loss'], test_metrics, 'final_test')
    wandb_cb.log_test({"Loss/Test": test_metrics['loss'], **test_metrics}, 'final_test')
    tensorboard_cb.log_confusion_matrix(test_confusion_matrix, 'final_test')
    wandb_cb.log_confusion_matrix(test_confusion_matrix, 'final_test')

    tensorboard_cb.close()
    wandb_cb.finish()