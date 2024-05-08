import argparse
import os
import sys
import logging
import torch
import time
from dataset import *
from utils import *
from model import Encoder
from loss import PointwiseRankingLoss, PairwiseRankingLoss, DeltaOrderLoss, RnCLoss
import yaml
from config.ordernamespace import OrderedNamespace
from main_linear import parse_regressor_option, train_regressor, set_logging
from exp_utils import compute_knn

print = logging.info


def parse_encoder_option():
    print("Parsing encoder's configurations")

    # General
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--config', type=str, help='path to your .yaml configuration file')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--ckpt', type=str, default='', help='path to the trained encoder')
    parser.add_argument('--seed', type=int, default=322)
    parser.add_argument('--save_folder', type=str, default=None)

    opt = parser.parse_args()

    # Load yaml file
    config = yaml.load(open(opt.config, "r"), Loader=yaml.FullLoader)
    opt = OrderedNamespace([opt, config])

    # Create model name
    loss_type = opt.Encoder.loss.loss_type 
    if "deltaorder" in loss_type:
        opt.model_name = '{}_{}_ep_{}_delta_{}'. \
            format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, opt.Encoder.loss.delta)
    elif loss_type == "RnC":
        opt.model_name = '{}_{}_ep_{}'. \
            format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs)
    elif loss_type == "pairwise":
        delta =  getattr(opt.Encoder.loss, "delta", 0.3)
        opt.model_name = '{}_{}_ep_{}_delta_{}_obj_{}'. \
            format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, delta, opt.Encoder.loss.objective) 
    elif loss_type  == "pointwise":
        opt.model_name = '{}_{}_ep_{}_norm_{}_obj_{}'. \
            format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, opt.Encoder.loss.feature_norm, opt.Encoder.loss.objective)  
    else:
        raise NotImplementedError(loss_type)

    if opt.data.noise_scale > 0.0:
        opt.model_name += f"_noise_scale_{opt.data.noise_scale}"

    opt.model_name += f"_trial_{opt.trial}"
        
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    # Create folders
    opt.model_path = f'./checkpoints/{loss_type}'
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder exist.')

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt

class AgeDB(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train', noise_scale=0.0):
        if split == "train" and noise_scale > 0:
            print(f"Adding gausain noise with scale {noise_scale} to the label.")
            df = pd.read_csv(f'./data/agedb_noise_scale_{noise_scale}.csv')
        else:
            df = pd.read_csv(f'./data/agedb_rank.csv')

        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)
        rank = np.asarray([row['rank']]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label, rank
    

def set_loader(opt):
    config = opt.data
    train_transform = get_transforms(split='train', aug=config.aug)
    val_transform = get_transforms(split='val', aug=config.aug)
    print(f"Train Transforms: {train_transform}")
    print(f"Val and Test Transforms: {train_transform}")

    train_dataset = AgeDB(
        data_folder=config.data_folder,
        transform=TwoCropTransform(train_transform),
        split='train',
        noise_scale=config.noise_scale,
    )
    val_dataset = AgeDB(
        data_folder=config.data_folder,
        transform=val_transform,
        split='val',
        noise_scale=config.noise_scale,
    )
    test_dataset = AgeDB(
        data_folder=config.data_folder,
        transform=val_transform,
        split='test',
        noise_scale=config.noise_scale,
    )
    print(f'Train set size: {train_dataset.__len__()}')
    print(f'Valid set size: {val_dataset.__len__()}')
    print(f'Test set size: {test_dataset.__len__()}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.Encoder.trainer.batch_size, shuffle=True,
        num_workers=opt.Encoder.trainer.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.Encoder.trainer.batch_size, shuffle=False,
        num_workers=opt.Encoder.trainer.num_workers, pin_memory=True, drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.Encoder.trainer.batch_size, shuffle=False,
        num_workers=opt.Encoder.trainer.num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


def set_model(opt):
    model = Encoder(name=opt.Encoder.type)
    config = opt.Encoder.loss
    if config.loss_type == "RnC":
        criterion = RnCLoss(delta=getattr(config, "delta", 0.0))
    elif config.loss_type == "pointwise":
        criterion = PointwiseRankingLoss(feature_norm=config.feature_norm, objective=config.objective)
    elif config.loss_type == "pairwise":
        criterion = PairwiseRankingLoss(delta=getattr(config, "delta", 0.3), objective=config.objective)
    else:
        criterion = DeltaOrderLoss(delta=config.delta)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    return model, criterion

def set_optimizer(opt, model, criterion):
    config = opt.Encoder.optimizer
    parameters = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate,
                                momentum=config.momentum, weight_decay=config.weight_decay)

    return optimizer

def save_model(model, optimizer, opt, epoch, save_file, criterion):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'criterion': criterion.state_dict()
    }
    torch.save(state, save_file)
    del state


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        images, labels, ranks = data_tuple
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            ranks = ranks.cuda(non_blocking=True)

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels, ranks)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.Encoder.trainer.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                       'feature_norm {avg_norms:.3f}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, avg_norms=features.norm(dim=1).mean()
            )
            print(to_print)
            sys.stdout.flush()

    if hasattr(criterion, "weight"):
        print(f"Learnable weight: {criterion.weight.exp().item()}")

    if hasattr(criterion, "bias"):
        print(f"Learnable bias: {criterion.bias.exp().item()}")

    if hasattr(criterion, "t"):
        print(f"Learnable t: {criterion.t.item()}")

def testing(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    Z = []
    y = []
    with torch.no_grad():
        for (images, labels, ranks) in val_loader:
            y.append(labels.numpy())
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            ranks = ranks.cuda(non_blocking=True)
            features = model(images)
            bsz = labels.shape[0]

            loss = criterion(features, labels, ranks)
            losses.update(loss.item(), bsz)
            Z.append(features.cpu().numpy())
            
    Z = np.concatenate(Z, axis=0)
    y = np.concatenate(y, axis=0)
    avg_nbr_diffs = compute_knn(Z, y)

    print(f"Test loss: {losses.avg:.3f}\t5 nearest neighbors label differences: {avg_nbr_diffs:.3f}")

    return losses.avg, avg_nbr_diffs

def train_encoder():
    opt = parse_encoder_option()
    regressor_opt = parse_regressor_option()
    regressor_opt.save_folder = opt.save_folder
    set_logging(regressor_opt)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model, criterion)

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    # training routine
    for epoch in range(start_epoch, opt.Encoder.trainer.epochs + 1):
        adjust_learning_rate(opt.Encoder, optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, opt)
        test_loss, avg_nbr_diffs = testing(test_loader, model, criterion)

        if epoch == 1:
            best_test_loss = test_loss
            best_avg_nbr_diffs = avg_nbr_diffs
            save_model(model, optimizer, opt, epoch, os.path.join(opt.save_folder, 'best_test_loss.pth'), criterion)
            save_model(model, optimizer, opt, epoch, os.path.join(opt.save_folder, 'best_nbr_ydiff.pth'), criterion)
        else:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                save_model(model, optimizer, opt, epoch, os.path.join(opt.save_folder, f'best_test_loss.pth'), criterion)
            if avg_nbr_diffs <= best_avg_nbr_diffs:
                best_avg_nbr_diffs = avg_nbr_diffs
                save_model(model, optimizer, opt, epoch, os.path.join(opt.save_folder, f'best_nbr_ydiff.pth'), criterion)

        # if epoch % opt.Encoder.trainer.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file, criterion)

        if epoch % opt.Encoder.trainer.save_curr_freq == 0:
            save_file = os.path.join(opt.save_folder, 'curr_last.pth')
            save_model(model, optimizer, opt, epoch, save_file, criterion)

        if epoch > 1 and epoch % opt.Encoder.trainer.test_regression_freq == 0:
            regressor_opt.ckpt = save_file
            train_regressor(regressor_opt)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.Encoder.trainer.epochs, save_file, criterion)

    # Train the linear regressor
    regressor_opt.ckpt = save_file
    train_regressor(regressor_opt)

    # Best test loss
    print(f"Best pretraining testing loss: {best_test_loss:.3f}")
    regressor_opt.ckpt = os.path.join(opt.save_folder, f'best_test_loss.pth')
    train_regressor(regressor_opt)

    # Best avg_nbr_diffs
    print(f"Best avg_nbr_diffs: {best_avg_nbr_diffs:.3f}")
    regressor_opt.ckpt = os.path.join(opt.save_folder, f'best_nbr_ydiff.pth')
    train_regressor(regressor_opt)


if __name__ == '__main__':
    train_encoder()
