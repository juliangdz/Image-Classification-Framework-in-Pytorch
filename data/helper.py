def get_sample_images(data_loader):
    images, labels = next(iter(data_loader))
    return images, labels
