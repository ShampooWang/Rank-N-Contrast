import argparse
import os
import sys
import logging
import torch
import time
from model import Encoder, model_dict
from utils import *
# Data
import pandas as pd
from PIL import Image
from torch.utils import data
# Others
import math
from tqdm import tqdm
from config.ordernamespace import OrderedNamespace
import yaml

print = logging.info


def parse_regressor_option():
    print("Parsing regressor's configurations")
    
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

    div_scale = getattr(opt.feature_extract, "div_scale", 1.0)
    normalized_by_D = getattr(opt.feature_extract, "normalized_by_D", False)
    opt.model_name = 'Regressor_{}_divScale_{}_normalied_D_{}_trial_{}'\
    .format(opt.data.dataset, div_scale, normalized_by_D, opt.trial)    
      
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    
    if opt.save_folder is None:
        opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1]) # default is saving regressor to the encoder ckpt's folder
    else:
        os.makedirs(opt.save_folder, exist_ok=True)

    
    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt

def set_logging(opt):
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, f'{opt.model_name}.log')),
            logging.StreamHandler()
        ]
    )

class AgeDB(data.Dataset):
    def __init__(self, opt, data_folder, transform=None, split='train', noise_scale=0.0, feature_dim=512, **kargs):
        if split == "train" and noise_scale > 0:
            print(f"Adding gausain noise with scale {noise_scale} to the label.")
            df = pd.read_csv(f'./data/agedb_noise_scale_{noise_scale}.csv')
        else:
            df = pd.read_csv(f'./data/agedb.csv')
    
        df = pd.read_csv(f'./data/agedb.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform

        if getattr(opt, "orthog_basis", False):
            from scipy.stats import ortho_group
            self.feature_basis = torch.from_numpy(ortho_group.rvs(dim=feature_dim, random_state=opt.seed)).float()
            q, r = divmod(len(self.df) , feature_dim)
            self.feature_basis = torch.cat([self.feature_basis]*q + [self.feature_basis[:r]], dim=0)
            assert self.feature_basis.shape[0] == len(self.df)
            print(f"Initialize feature basis with rank: {feature_dim}, feature basis shape: {self.feature_basis.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        returned_items = {
            "img": img,
            "label": label,
        }

        if hasattr(self, "feature_basis"):
            returned_items["feature_basis"] = self.feature_basis[index]
        
        return returned_items
    
class AgeDBFeatureLabel_pairs(data.Dataset):
    def __init__(self, features, labels) -> None:
        if isinstance(features, list):
            self.features = torch.cat(features, dim=0)
        else:
            self.features = features

        if isinstance(features, list):
            self.labels = torch.cat(labels, dim=0)
        else:
            self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def get_encoder_features(opt, dataset, model):
    config = opt.Regressor.trainer
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False
    )
    div_scale = getattr(opt.feature_extract, "div_scale", 1.0)
    print(f"Start extracting frozen features. Divide feature by the scale: {div_scale}")
    normalized_by_D = getattr(opt.feature_extract, "normalized_by_D", False)
    if normalized_by_D: 
        print(f"Normalize features by sqrt(D)")
        
    model.eval()
    Z = []
    y = []
    with torch.no_grad():
        for returned_items in tqdm(loader):
            images = returned_items["img"].cuda(non_blocking=True)
            features = model(images) / div_scale
            if normalized_by_D:
                features = features / math.sqrt(features.shape[1])
            Z.append(features.cpu())
            y.append(returned_items["label"])
    del loader
    return AgeDBFeatureLabel_pairs(Z, y)


def set_loader(opt, model):
    train_transform = get_transforms(split='train', aug=opt.data.aug)
    val_transform = get_transforms(split='val', aug=opt.data.aug)
    train_dataset = globals()[opt.data.dataset](opt, transform=train_transform, split='train', **opt.data)
    val_dataset = globals()[opt.data.dataset](opt, transform=val_transform, split='val', **opt.data)
    test_dataset = globals()[opt.data.dataset](opt, transform=val_transform, split='test', **opt.data)

    if opt.Regressor.trainer.verbose:
        print(f"Train Transforms: {train_transform}")
        print(f"Val Transforms: {val_transform}")
        print(f'Train set size: {train_dataset.__len__()}\t'
            f'Val set size: {val_dataset.__len__()}\t'
            f'Test set size: {test_dataset.__len__()}')

    trainFeatureLabel_loader = torch.utils.data.DataLoader(
        get_encoder_features(opt, train_dataset, model), batch_size=opt.Regressor.trainer.batch_size, shuffle=True, num_workers=opt.Regressor.trainer.num_workers, pin_memory=True
    )
    valFeatureLabel_loader = torch.utils.data.DataLoader(
        get_encoder_features(opt, val_dataset, model), batch_size=opt.Regressor.trainer.batch_size, shuffle=False, num_workers=opt.Regressor.trainer.num_workers, pin_memory=True
    )
    testFeatureLabel_loader = torch.utils.data.DataLoader(
        get_encoder_features(opt, test_dataset, model), batch_size=opt.Regressor.trainer.batch_size, shuffle=False, num_workers=opt.Regressor.trainer.num_workers, pin_memory=True
    )

    return trainFeatureLabel_loader, valFeatureLabel_loader, testFeatureLabel_loader


def set_model(opt):
    model = Encoder(name=opt.Encoder.type)
    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()

    dim_in = model_dict[opt.Encoder.type][1]
    dim_out = get_label_dim(opt.data.dataset)
    regressor = torch.nn.Linear(dim_in, dim_out, bias=getattr(opt.Regressor, "bias", True))
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    model = model.cuda()
    regressor = regressor.cuda()
    criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict)
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    return model, regressor, criterion


def train_epoch(loader, regressor, criterion, optimizer, epoch, opt):
    regressor.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (features, labels) in enumerate(loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        output = regressor(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if (idx + 1) % opt.Regressor.trainer.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
        #         epoch, idx + 1, len(loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses))
        #     sys.stdout.flush()

    if opt.Regressor.trainer.verbose:
        print('Epoch [{0}], average loss: {1.avg:.3f}'.format(epoch, losses))
    sys.stdout.flush()


def validate(loader, regressor):
    regressor.eval()

    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for features, labels in loader:
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            output = regressor(features)
            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

    return losses.avg


def train(
        opt, 
        criterion,
        train_loader, 
        val_loader, 
        regressor
    ):

    # build optimizer
    optimizer = set_optimizer(opt.Regressor, regressor)

    save_file_best = os.path.join(opt.save_folder, f"{opt.model_name}_best.pth")
    save_file_last = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
    best_error = 1e5

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        regressor.load_state_dict(ckpt_state['state_dict'])
        start_epoch = ckpt_state['epoch'] + 1
        best_error = ckpt_state['best_error']
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")


    epoch_counter = range(start_epoch, opt.Regressor.trainer.epochs + 1)
    if not opt.Regressor.trainer.verbose:
        epoch_counter = tqdm(epoch_counter)

    print("Start training regressor")
    for epoch in epoch_counter:
        adjust_learning_rate(opt.Regressor, optimizer, epoch)

        # train for one epoch
        train_epoch(train_loader, regressor, criterion, optimizer, epoch, opt)

        valid_error = validate(val_loader, regressor)
        is_best = valid_error < best_error
        best_error = min(valid_error, best_error)

        if opt.Regressor.trainer.verbose:
            print('Val L1 error: {:.3f}'.format(valid_error))
            print(f"Best Error: {best_error:.3f}")

        if is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': regressor.state_dict(),
                'best_error': best_error
            }, save_file_best)

        torch.save({
            'epoch': epoch,
            'state_dict': regressor.state_dict(),
            'last_error': valid_error
        }, save_file_last)

    return save_file_best


def train_regressor(opt=None):
    if opt is None:
        opt = parse_regressor_option()

    # Fixing random seed    
    set_seed(opt.seed)

    # build model and criterion
    model, regressor, criterion = set_model(opt)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt, model)

    # training routine
    save_file_best = train(
        opt,
        criterion,
        train_loader,
        val_loader,
        regressor
    )

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    regressor.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
    test_loss = validate(test_loader, regressor)
    to_print = 'Test L1 error: {:.3f}'.format(test_loss)
    print(to_print)


if __name__ == '__main__':
    opt = parse_regressor_option()
    train_regressor(opt)
