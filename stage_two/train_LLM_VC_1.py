import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from torch.distributions import Categorical
from tqdm import  tqdm
import os
import copy
import time
import gc
import random
import numpy as np
import argparse
#from sklearn.metrics import confusion_matrix
from pysot.utils.model_load import load_pretrain, load_network
#import models.models as models
import models.LLM_models as models
from util.utils import *
from util.optimization import AdamW
from LLM_VC_dataset import LLM_dataset
import pickle

#from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor



parser = argparse.ArgumentParser(description='LLM_VC for sports')
parser.add_argument('--video_feature_root', default='/home/xzy/xzy_nba/VG_NBA_Timesformer-features', type=str, help='npy features root')
parser.add_argument('--tokenizer_name', default="/home/xzy/xzy_nba/meta-llama/Llama-3.2-3B", type=str, help='tokenizer name')
# train
parser.add_argument('--train_info_root', default="/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/train_video_info.json", type=str, help='train json file root')
parser.add_argument('--train_top2_root', default="/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/D_train_videoid_player_feature_top2.pkl", type=str, help='train pkl file root')
parser.add_argument('--train_batch_size', default=8, type=int, help='train batch size')  # 1B--32    3B--8
parser.add_argument('--train_num_workers', default=8, type=int, help='train number workers')
# test
parser.add_argument('--test_info_root', default="/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/test_video_info.json", type=str, help='test json file root')
parser.add_argument('--test_top2_root', default="/home/xzy/xzy_nba/LLM_VC/Player_identify/code/A_data_preprocessing/E_videoid_top2.pkl", type=str, help='test pkl file root')
parser.add_argument('--test_batch_size', default=8, type=int, help='test batch size')
parser.add_argument('--test_num_workers', default=8, type=int, help='test number workers')
# training setting
parser.add_argument('--random_seed', default=42, type=int, help='random seed for reproduction')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')  # 3B--7e-6  1B--5e-5
parser.add_argument('--num_epoch', default=100, type=int, help='epoch number')
parser.add_argument('--num_video_query_token', default=32, type=int, help='query tokens number for Qformer')
parser.add_argument('--feature_dim', default=768, type=int, help='feature dim')
parser.add_argument('--save_root', default='./without_prompt_LLM_3B_ckpt', type=str, help='save root')
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
# Training parameters & choose
parser.add_argument('--max_lr', default=1e-5, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=35, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')

# If continue training from any epoch  # ***
parser.add_argument("--continue_train", type=bool, default=False)
parser.add_argument("--pre_max_CIDEr", type=float, default=0.0)
parser.add_argument("--pre_epoch", type=int, default=0)
parser.add_argument("--load_ckpt", type=str, default="./LLM_ckpt/model_save_best_val_CIDEr.pth")

# ori load model
parser.add_argument('--load_model', default=False, action='store_true', help='load model')
parser.add_argument('--model_path', default="/home/shige4090/xzy/BART_VC/result/[nba]_DFGAR_<2023-11-14_00-42-36>/epoch41.pth", type=str, help='pretrained model path')

# dataloader
parser.add_argument('--num_frame', default=60, type=int, help='number of frames for each clip')
parser.add_argument('--max_words', default=30, type=int, help='number of frames for each clip')

args = parser.parse_args()

#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Use CIDEr score to do validation
def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict =dict()
    gt_captions_dict = dict()
    for i, caption in enumerate(predicted_captions):
        predicted_captions_dict[i] = [caption]
    for i, caption in enumerate(gt_captions):
        gt_captions_dict[i] = [caption]
    #print("predicted_captions_dict: ", predicted_captions_dict, flush=True)
    #print("gt_captions_dict: ", gt_captions_dict, flush=True)
    _, cider_scores = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()


##################
################## main
##################
def main():
    global args

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# class LLM_dataset(Dataset):
#     def __init__(self, feature_root, video_info_path, videoid_top2_path, max_frames, max_words,
#                  tokenizer_name='meta-llama/Meta-Llama-3-8B', max_token_length=128)
# load dataset
    train_set = LLM_dataset(feature_root=args.video_feature_root, video_info_path=args.train_info_root, videoid_top2_path=args.train_top2_root, max_frames=args.num_frame, max_words=args.max_words, tokenizer_name=args.tokenizer_name)
    test_set = LLM_dataset(feature_root=args.video_feature_root, video_info_path=args.test_info_root, videoid_top2_path=args.test_top2_root, max_frames=args.num_frame, max_words=args.max_words, tokenizer_name=args.tokenizer_name)

    train_loader = data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("%d train samples and %d test samples" % (len(train_loader)*args.train_batch_size, len(test_loader)*args.test_batch_size))
    model = models.LLM_Captioner(llm_ckpt=args.tokenizer_name, tokenizer_ckpt=args.tokenizer_name, num_video_query_token=args.num_video_query_token,
                             num_features=args.feature_dim, device=args.device)
    #print("model-names: ", model)
    # model = torch.nn.DataParallel(model).cuda()
    # model = torch.nn.DataParallel(model, device_ids=None).to('cuda')
    # model = model.to('cuda')
    if args.continue_train:
        model.load_state_dict(torch.load(args.load_ckpt))
    optimizer = AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.save_root, exist_ok=True)
    max_val_CIDEr = max(float(0), args.pre_max_CIDEr)




    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(args.save_root, '--------------------Number of parameters--------------------')
    print_log(args.save_root, parameters)

    # #组合
    # if hasattr(model, 'module'):
    #     model = model.module


    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch'] + 1
        #model = load_pretrain(model, args.model_path)
        #model_class = load_pretrain(model_class, args.bert_class_path)
        start_epoch = 42
    else:
        start_epoch = 1

    # training phase
    for epoch in range(start_epoch, args.num_epoch + 1):
        torch.cuda.empty_cache()
        #print('----- %s at epoch #%d' % ("Train", epoch))
        print_log(args.save_root, '----- %s at epoch #%d' % ("Train", epoch))
        train_log = train(train_loader, model, optimizer, epoch)
        #train_log = train(train_loader, model, criterion, optimizer, epoch)
        print_log(args.save_root, 'Loss: %.4f, Using %.1f seconds' %
                  (train_log['loss'], train_log['time']))


        if epoch % args.test_freq == 0:
            torch.cuda.empty_cache()
            #print('----- %s at epoch #%d' % ("Test", epoch))
            print_log(args.save_root, '----- %s at epoch #%d' % ("Test", epoch))
            test_log = validate(test_loader, model, epoch)
            print_log(args.save_root, 'C: %.4f, M: %.4f, R: %.4f, B_1: %.4f, B_2: %.4f, B_3: %.4f, B_4: %.4f, p_acc: %.4f, Using %.1f seconds' %
                      (test_log['C'], test_log['M'], test_log['R'], test_log['B_1'], test_log['B_2'], test_log['B_3'], test_log['B_4'], test_log['p_acc'], test_log['time']))
            #
            print_log(args.save_root, '---------- Save at epoch #%d.' %
                      (test_log['epoch']))
            # print_log(save_path, '----------Best MPCA: %.2f%% at epoch #%d.' %
            #           (test_log['best_mpca'], test_log['best_mpca_epoch']))

            #if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                #'scheduler': scheduler.state_dict(),
            }
            result_path = args.save_root + '/epoch%d.pth' % (epoch)
            torch.save(state, result_path)

# train_loader, model, optimizer, epoch
def train(train_loader, model, optimizer, epoch):
#def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    epoch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()
    # print(accuracies)

    # switch to train mode
    model.train()
    train_loss_accum = 0.0
# for i, (video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature) in enumerate(train_loader):
# for i, (images, activities) in enumerate(tqdm(train_loader, total=7624, position=0)):
    for i, (video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, gt_caption) in enumerate(tqdm(train_loader, total=8668, position=0)):
        # video = video.cuda()
        # video_mask = video_mask.cuda()
        # caption_tokens = caption_tokens.cuda()
        # labels = labels.cuda()
        # attention_mask = attention_mask.cuda()
        # name_one_embeds = name_one_embeds.cuda()
        # entity_one_feature = entity_one_feature.cuda()
        # name_two_embeds = name_two_embeds.cuda()
        # entity_two_feature = entity_two_feature.cuda()

        loss = model(video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, validating=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)
    print("avg_train_loss: ", avg_train_loss)
    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': avg_train_loss,
        #'group_acc': accuracies.avg * 100.0,
    }
    return train_log


@torch.no_grad()
def validate(test_loader, model, epoch):
    # global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    epoch_timer = Timer()
    # losses = AverageMeter()
    # accuracies = AverageMeter()
    # switch to eval mode
    model.eval()
    val_CIDEr = 0.0
    res = {}
    gts = {}
    eval_id = 0
    p_true = []
    p_pred = []
    #for i, (images, activities, text_nba) in enumerate(test_loader):
    #for i, (images, activities, text_nba, image_clip) in enumerate(tqdm(test_loader, total=1548, position=0)):
    #for i, (images, activities, text_nba, game_players, names_list) in enumerate(test_loader):
    for i, (video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, gt_caption) in enumerate(tqdm(test_loader, total=1060, position=0)):
        # video = video.cuda()
        # video_mask = video_mask.cuda()
        # caption_tokens = caption_tokens.cuda()
        # labels = labels.cuda()
        # attention_mask = attention_mask.cuda()
        # name_one_embeds = name_one_embeds.cuda()
        # entity_one_feature = entity_one_feature.cuda()
        # name_two_embeds = name_two_embeds.cuda()
        # entity_two_feature = entity_two_feature.cuda()

        res_text = model(video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, validating=True)
        #print("res_text: ", res_text)
        #print("gt_text: ", gt_caption)
        for i, _ in enumerate(res_text):
            res[eval_id] = [res_text[i]]  # 预测
            gts[eval_id] = [gt_caption[i]]
            eval_id = eval_id + 1
        # # true = true + activities_in.tolist()
        # # pred = pred + torch.argmax(score, dim=1).tolist()
        # #
        # # # calculate loss
        # #loss = criterion(outputs, hint_encoded)
        #
        # #
        # # # measure accuracy and record loss
        # # group_acc = accuracy(score, activities_in)
        # losses.update(loss, num_batch)
        # accuracies.update(group_acc, num_batch)
    avg_bleu_score, bleu_scores = Bleu(4).compute_score(gts, res)
    avg_cider_score, cider_scores = Cider().compute_score(gts, res)
    avg_meteor_score, meteor_scores = Meteor().compute_score(gts, res)
    avg_rouge_score, rouge_scores = Rouge().compute_score(gts, res)
    p_score = p_acc(p_true, p_pred)
    p_new = (p_score * 4 + avg_cider_score * 2 + avg_meteor_score * 2 + avg_rouge_score) / 9
    #print('C, M, R, B:', avg_cider_score, avg_meteor_score, avg_rouge_score, avg_bleu_score)
    test_log = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'C': avg_cider_score,
        'M': avg_meteor_score,
        'R': avg_rouge_score,
        'B_1': avg_bleu_score[0],
        'B_2': avg_bleu_score[1],
        'B_3': avg_bleu_score[2],
        'B_4': avg_bleu_score[3],
        'p_acc': p_new
    }
    #
    return test_log


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # print(val.shape)
        # print(n)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    # print(target.shape)
    # print(output.shape)
    #print(output.shape[0])
    correct = torch.sum(torch.eq(target, output)).float()
    # print("correct", correct)
    # print("fenmu", output.shape[0])
    return correct.item() / output.shape[0]
def p_acc(p_true, p_pred):
    p_true = np.array(p_true)
    p_pred = np.array(p_pred)
    assert len(p_true) == len(p_pred), "size not equal"
    return np.sum(p_true == p_pred) / len(p_true)

if __name__ == '__main__':
    main()
