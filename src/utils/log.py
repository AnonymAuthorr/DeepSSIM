from torch.utils.tensorboard import SummaryWriter

# This function logs the loss and performance value for visualization in TensorBoard.
# It also prints this value to console.
# Author: Author1
    
def log_metrics(path: str, mode: str, epoch: int, loss_mean: float, loss_std: float, mae: float) -> None:
    with SummaryWriter(path) as writer:
        writer.add_scalar(mode + '/loss_mean', loss_mean, epoch)
        writer.add_scalar(mode + '/loss_std', loss_std, epoch)
        writer.add_scalar(mode + '/mae', mae, epoch)
        print('Loss [MSE] =', loss_mean)
        print('Standard Deviation [MSE] =', loss_std)
        print('Performance [MAE] =', mae)