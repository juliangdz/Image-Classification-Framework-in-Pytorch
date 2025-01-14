import torch

def get_sample_images(data_loader):
    images, labels = next(iter(data_loader))
    return images, labels

def get_balanced_sample(data_loader, n_pos=32, n_neg=32):
    """
    Iterates through the data_loader until it collects 
    n_pos positives (label=1) and n_neg negatives (label=0).
    Returns a tuple of (images, labels) of shape (n_pos + n_neg, ...).
    """
    pos_images = []
    pos_labels = []
    neg_images = []
    neg_labels = []

    # Collect enough positive and negative samples
    for images, labels in data_loader:
        # Find indices for positive and negative
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        # Add positive samples from the batch
        if pos_mask.any():
            pos_images.append(images[pos_mask])
            pos_labels.append(labels[pos_mask])

        # Add negative samples from the batch
        if neg_mask.any():
            neg_images.append(images[neg_mask])
            neg_labels.append(labels[neg_mask])

        # Count how many we've collected so far
        total_pos = sum(img.shape[0] for img in pos_images)
        total_neg = sum(img.shape[0] for img in neg_images)

        # Stop early if we already have enough
        if total_pos >= n_pos and total_neg >= n_neg:
            break

    # If we didn't collect enough, raise an exception or handle it
    if total_pos < n_pos or total_neg < n_neg:
        raise ValueError(
            f"Not enough samples collected. Needed {n_pos} pos and {n_neg} neg, "
            f"got {total_pos} pos and {total_neg} neg."
        )

    # Concatenate all the pos/neg into single tensors
    pos_images = torch.cat(pos_images, dim=0)[:n_pos]
    pos_labels = torch.cat(pos_labels, dim=0)[:n_pos]
    neg_images = torch.cat(neg_images, dim=0)[:n_neg]
    neg_labels = torch.cat(neg_labels, dim=0)[:n_neg]

    # Combine positives and negatives
    balanced_images = torch.cat([pos_images, neg_images], dim=0)
    balanced_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # (Optional) Shuffle this balanced set
    # indices = torch.randperm(balanced_images.size(0))
    # balanced_images = balanced_images[indices]
    # balanced_labels = balanced_labels[indices]

    return balanced_images, balanced_labels