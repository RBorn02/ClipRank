import torch
import torch.nn as nn
import torch.optim as optim

import clip

from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import os
import argparse
import yaml
import pandas as pd

from data import get_contrastive_dataloader, RankingLoader
from trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train ClipRank')
    parser.add_argument('data_train', help='path to training data')
    parser.add_argument('anno_file', help='path to annotation file')
    #parser.add_argument('vocab_file', help='path to vocab file')
    parser.add_argument('--model', default='RN50', help='vision encoder model',
                        choices=['RN59', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'
                                 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--device', default='cuda', help='which device to use for training')
    parser.add_argument('--lr', default=5*1e-6, type=float, help='learning rate set for resnet50')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size for training')
    parser.add_argument('--wd', default=0.2, type=float, help='weight decay for adam')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta 1 for adam')
    parser.add_argument('--beta2', default=0.98, type=float, help='beta 2 for adam')
    parser.add_argument('--epsilon', default=1e-6, type=float, help='adam epsilon')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--workers', default=64, type=int, help='number of workers for dataloader')
    parser.add_argument('--ranking', default=False, action='store_true', help='use ranking as a training task')
    parser.add_argument('--num_trans', default=4, type=int, help='number of image transformations for ranking')
    parser.add_argument('--lmbda', default=0.8, type=int, help='scaling for ranking and contrastive loss')
    parser.add_argument('--ranking_batch', default=124, type=int, help='batch size for ranking task')
    parser.add_argument('--scale', default=12, type=int, help='scale parameter for ranking loss')
    parser.add_argument('--alpha', default=0.05, type=float, help='margin for listwise loss')
    parser.add_argument('--beta', default=0.5, type=float, help='margin for positive loss')
    parser.add_argument('--gamma', default=1.0, type=float, help='scale for positive and listloss')


    args = parser.parse_args()

    writer = SummaryWriter()
    save_config_file(writer.log_dir, args)
    
    #Convert Weights to FP32 for Finetuning https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float()
    
    #Load and Initialize Model
    model, preprocess = clip.load(args.model, device=args.device)

    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if torch.cuda.device_count() > 1:
        print("We have", torch.cuda.device_count(), "GPUs available!")
        model = nn.DataParallel(model, device_ids=[range(torch.cuda.device_count())])
    
    print('Loading {} model'.format(args.model))
    
    #Initialize Optimizer
    optimizer = optim.Adam([
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},], 
                           lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.wd, eps=args.epsilon)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    #Get Dataloaders ###TODO: Implement val_loader
    train_loader, val_loader = get_contrastive_dataloader(args, preprocess)

    loader = RankingLoader()
    ranking_loader = loader.get_ranking_loader(args, model.visual.input_resolution)

    trainer = Trainer(model, optimizer, scheduler, args)

    test_i_top1_ls = []
    test_i_top5_ls = []
    test_t_top1_ls = []
    test_t_top5_ls = []

    results = {'train_loss': [],
               'test_R@1_Image': test_i_top1_ls,
               'test_R@5_Image': test_i_top5_ls,
               'test_R@1_Text': test_t_top1_ls,
               'test_R@5_Text': test_t_top5_ls
              }
    
    save_name_pre = '{}_{}_{}_{}_{}'.format(
        args.model, args.wd,
        args.batch_size, args.lr,  args.epochs)
    csv_dir = os.path.join(writer.log_dir, '{}_stats.csv'.format(save_name_pre))
    model_dir = os.path.join(writer.log_dir, '{}_model.pth'.format(save_name_pre))
    final_model_dir = os.path.join(writer.log_dir, '{}_final_model.pth'.format(save_name_pre))
    fig_dir = os.path.join(writer.log_dir, '{}_loss_acc.png'.format(save_name_pre))
    
    for epoch in range(1, args.epochs+1):
        loss = trainer.train(train_loader, ranking_loader, epoch)
        
        results['train_loss'].append(loss.cpu().detach())
        print(results['train_loss'])
        writer.add_scalar('loss/train', results['train_loss'][-1], epoch)
        
        image_recall, text_recall = trainer.validate(val_loader, epoch)
        test_i_top1_ls.append(image_recall[0].cpu())
        test_i_top5_ls.append(image_recall[1].cpu())
        test_t_top1_ls.append(text_recall[0].cpu())
        test_t_top5_ls.append(text_recall[1].cpu())

        writer.add_scalar('IR@1', results['test_R@1_Image'][-1], epoch)
        writer.add_scalar('IR@5', results['test_R@5_Image'][-1], epoch)
        writer.add_scalar('TR@1', results['test_R@1_Text'][-1], epoch)
        writer.add_scalar('TR@5', results['test_R@5_Text'][-1], epoch)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(csv_dir, index_label='epoch')
        
        if image_recall[0] > best_acc:
            best_acc = image_recall[0]
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, model_dir)

        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, final_model_dir)
        
    # plotting loss and accuracies
    df = pd.read_csv(csv_dir)
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20,10))
    axes[0, 0].set_title('Loss/Train')
    axes[0, 1].set_title('IR@1/test')
    axes[1, 1].set_title('TR@1/test')
    sns.lineplot(ax=axes[0, 0], x="epoch", y="train_loss", data=df)
    sns.lineplot(ax=axes[0, 1], x="epoch", y="test_R@1_Image", data=df)
    sns.lineplot(ax=axes[1, 1], x="epoch", y="test_R@1_Text", data=df)        
    fig.savefig(fig_dir)


    
