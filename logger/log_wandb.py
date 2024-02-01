import wandb
import torchvision.transforms as transforms

class WandBCallback:
    def __init__(self,config:dict):
        project = config.get('project','Pytorch_MNIST_Classification')
        wandb.login()
        wandb.init(project=project)

    def log(self, data, step):
        wandb.log(data, step=step)

    def log_images(self, images, labels, tag='samples'):
        images = [wandb.Image(image, caption=str(label)) for image, label in zip(images, labels)]
        wandb.log({tag: images}, step=0)
        
    def log_confusion_matrix(self, matrix, step):
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(matrix, step=step)})
        
    def log_evaluation_images(self, images, predicted_labels, true_labels, tag='Eval Samples', step=0):
        images = [transforms.functional.to_pil_image(image) for image in images]
        logged_images = [wandb.Image(image, caption=f'Pred: {pred}, True: {true}') 
                         for image, pred, true in zip(images, predicted_labels, true_labels)]
        wandb.log({tag: logged_images}, step=step)

    def log_test(self, data, step):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()