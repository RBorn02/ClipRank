import torch
from tqdm import tqdm
from losses import ContrastiveLoss, RankingLoss
import clip

import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Trainer:
    def __init__(self, model, optimizer, scheduler, args):
        super(Trainer, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.con_loss = ContrastiveLoss()
        self.rank_loss = RankingLoss(args)

        self.device = args.device
        self.epochs = args.epochs
        self.num_trans = args.num_trans
        self.batch_size = args.batch_size
        self.ranking_batch = args.ranking_batch
        self.lmbda = args.lmbda
        self.ranking = args.ranking

    def train(self, train_loader, ranking_loader, epoch):
        self.model.train()
        n_px = self.model.visual.input_resolution

        train_bar = tqdm(train_loader)

        contrastive_loss = []
        ranking_loss = []
        losses = []

        start = time.time()

        if self.ranking:
            for train_tuple, ranking_tuple in zip(train_bar, ranking_loader):
                dtime = time.time() - start

                image, text = train_tuple[0].to(self.device), train_tuple[1].to(self.device)
                image_logits, text_logits = self.model(image, text)
                con_loss = self.con_loss(image_logits, text_logits)
                contrastive_loss.append(con_loss)

                image, text = ranking_tuple[0].to(self.device), ranking_tuple[1].to(self.device)
                image_features = self.model.encode_image(image.reshape(-1, 3, n_px, n_px))
                text_features = self.model.encode_text(text)
                rank_loss = self.rank_loss(image_features.reshape(self.num_trans+1, self.ranking_batch, -1), text_features)
                ranking_loss.append(rank_loss)

                loss = con_loss + rank_loss * self.lmbda
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()

                convert_models_to_fp32(self.model) 
                self.optimizer.step()
                convert_models_to_mix(self.model)

                ctime = time.time() - start - dtime
                start = time.time()

                train_bar.set_description(
                    '{}Train{} {}Epoch:{}[{}/{}] {} Loss:{}{:.4f} Contrastive Loss:{:.4f} Ranking Loss:{:.4f} ctime:{:.6f} dtime:{:.6f}]'
                    .format(
                        bcolors.OKCYAN, bcolors.ENDC,
                        bcolors.WARNING, bcolors.ENDC,
                        epoch,
                        self.epochs,
                        bcolors.WARNING, bcolors.ENDC,
                        sum(losses) / len(losses),
                        sum(contrastive_loss) / len(contrastive_loss),
                        sum(ranking_loss) / len(ranking_loss),
                        ctime,
                        dtime,
                        bcolors.WARNING, bcolors.ENDC))
        
            self.scheduler.step()

            return sum(losses) / len(losses)

        else:
            for image, text in train_bar:
                dtime = time.time() - start

                image, text = image.to(self.device), text.to(self.device)
                image_logits, text_logits = self.model(image, text)
                con_loss = self.con_loss(image_logits, text_logits)
                contrastive_loss.append(con_loss)

                self.optimizer.zero_grad()
                con_loss.backward()

                convert_models_to_fp32(self.model) 
                self.optimizer.step()
                convert_models_to_mix(self.model)

                ctime = time.time() - start - dtime
                start = time.time()

                train_bar.set_description(
                    '{}Train{} {}Epoch:{}[{}/{}] {} Loss:{}{:.4f} ctime:{:.6f} dtime:{:.6f}]'
                    .format(
                        bcolors.OKCYAN, bcolors.ENDC,
                        bcolors.WARNING, bcolors.ENDC,
                        epoch,
                        self.epochs,
                        bcolors.WARNING, bcolors.ENDC,
                        sum(contrastive_loss) / len(contrastive_loss),
                        ctime,
                        dtime,
                        bcolors.WARNING, bcolors.ENDC))
        
            self.scheduler.step()

            return sum(contrastive_loss) / len(contrastive_loss)

    def validate(self, val_loader, epoch):
        self.model.eval()

        val_bar = tqdm(val_loader)

        R1, R5, tot_num = [], [], 0.0

        start = time.time()
        for image, text in val_bar:
            dtime = time.time() - start

            image, text = image.to(self.device), text.to(self.device)
            with torch.no_grad():
                image_logits, text_logits = self.model(image, text)

            recall_list = compute_recall(image_logits, [1, 5])
            

            R1.append(recall_list[0])
            R5.append(recall_list[1])

            ctime = time.time() - start - dtime
            start = time.time()
            val_bar.set_description(
                '{}Eval{} {}Epoch:{}[{}/{}] {}Recall@1:{}{:.4f} {}Recall@5:{}{:.4f} ctime:{:.6f} dtime:{:.6f}]'
                .format(
                    bcolors.OKCYAN, bcolors.ENDC,
                    bcolors.WARNING, bcolors.ENDC,
                    epoch,
                    self.epochs,
                    bcolors.WARNING, bcolors.ENDC,
                    sum(R1) / len(R1),
                    bcolors.WARNING, bcolors.ENDC,
                    sum(R5) / len(R5),
                    ctime,
                    dtime,
                    bcolors.WARNING, bcolors.ENDC))   
        return (sum(R1) / len(R1)), (sum(R5) / len(R5))


def compute_recall(image_logits, k):
    labels = torch.arange(image_logits.shape[0]).to(image_logits.device)

    _, pred_labels = torch.topk(image_logits, k[-1], dim=1)

    recall_list = []
    for i in k:
        correct_bool = labels.reshape(-1, 1).repeat(1, pred_labels.shape[1]) == pred_labels[:,:i]
        correct_vec = torch.any(correct_bool, dim=1)
        recall = torch.mean(correct_vec.float())
        recall_list.append(recall)
    return recall_list

def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float()

def convert_models_to_mix(model):
    clip.model.convert_weights(model)




        
