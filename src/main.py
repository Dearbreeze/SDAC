import torch
import numpy as np
import cv2
from tqdm import tqdm
import State as State
from pixelwise_a3c import *
from Glimpse import *
from SDAC_Network import *
import matplotlib.pyplot as plt
import torch.optim as optim
from Dataloader_Hecktor_npy import *
# from Dataloader_Brain import *
import argparse
import Visualizer
import Mean_vail,Sum_vail
from medpy.metric import binary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(3407)
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

parser = argparse.ArgumentParser()
# hyper-parameters
parser.add_argument('--BATCH_SIZE', type = int, default = 1 , help = 'batch size')
parser.add_argument('--LR', type = float, default = 3e-4, help='learning rate')
parser.add_argument('--DIS_LR', type = int, default = 3e-4, help = '')
parser.add_argument('--visual_iter', type=int, default= 3, help='Visualization every X iterations')
parser.add_argument('--GAMMA', type = float, default = 0.95, help='GAMMA')
parser.add_argument('--MAX_EPISODE', type = int, default = 1000, help = 'interval of assigning the target network parameters to evaluate network')
parser.add_argument('--NUM_ACTIONS', type = int, default = 2, help = 'number of actions')
parser.add_argument('--Data_root', type = str, default = '/mnt/sdd/lyx/lyx_dataset/Brain', help = 'path of dataset')
parser.add_argument('--MOVE_RANGE', type = int, default = 2, help = 'range of actions')
parser.add_argument('--EPISODE_LEN', type = int, default = 10, help = '')
parser.add_argument('--img_size', type = int, default = 256, help = 'img size')
parser.add_argument('--box_size', type = int, default = 144, help = '')
parser.add_argument('--save_path', type = str, default = './checkpoints/', help = 'path of model save')
parser.add_argument('--pre_train', type = str, default = None,)
parser.add_argument('--name', type = str, default = 'SDAC_', help = 'name')


def main(opt, vis):
    same_seeds(1)
    model = SDAC(num_classes = 2).to(device)
    if opt.pre_train is not None:
        print('============== Loading Pre-train Model ==================')
        model.load_state_dict(torch.load(opt.pre_train),)
        # print(model)
    optimizer = optim.Adam(model.parameters(), lr=opt.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    path = opt.Data_root


    train_img_list, train_anno_list, vail_img_list, vail_anno_list, = PathList(path)

    TrainDataset = MakeDataset(path+'/Train', train_img_list, train_anno_list, True, opt.img_size)
    VailDataset = MakeDataset(path+'/Test', vail_img_list, vail_anno_list, False)

    print('The number of training data: ' + str(len(train_img_list)))
    print('The number of vailing data: ' + str(len(vail_anno_list)))

    TrainLoader = DataLoader(TrainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=8)
    VailLoader = DataLoader(VailDataset, batch_size=1, shuffle=False, num_workers=8)

    current_state = State.State((opt.BATCH_SIZE, 1, opt.img_size, opt.img_size), opt.MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(opt, model, optimizer, opt.BATCH_SIZE, opt.EPISODE_LEN, opt.GAMMA)
    Kc_Glimpse = Det_State(opt, model, optimizer, opt.BATCH_SIZE)

    for n_epi in tqdm(range(0, opt.MAX_EPISODE), ncols=70):

        for batch_idx, (state, targets, _) in enumerate(tqdm(TrainLoader)):
            # break
            l = state.numpy()
            gt = targets.numpy()
            state = Kc_Glimpse.step(state)
            Kc_Glimpse.train(True)
            # print(l.shape,type(l))
            # raw_n = np.random.normal(0, opt.sigma, (opt.BATCH_SIZE, 1, opt.img_size, opt.img_size)).astype(l.dtype) / 255
            current_state.reset(state.numpy())
            reward = np.zeros(l.shape, l.dtype)
            sum_reward = 0
            if n_epi % opt.visual_iter == 0:
                image = np.asanyarray(l[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
                image = np.squeeze(image)
                vis.img('ori_img', (torch.from_numpy(l).data.cpu()[0]))
                vis.img('GT', (torch.from_numpy(gt*l).data.cpu()[0]))
            for t in range(opt.EPISODE_LEN):
                if n_epi % opt.visual_iter == 0:

                    vis.img('img', (torch.from_numpy(current_state.image).data.cpu()[0]))
                previous_image = np.clip(current_state.image.copy(), a_min=0., a_max=1.)

                action, action_prob = agent.act_and_train(current_state.image, reward)
                # print(action.size())  torch.Size([1, 256, 256])
                if n_epi % opt.visual_iter == 0:

                    vis.heatmap(action[0].data.squeeze(0), opts=dict(xmin=0, xmax=opt.NUM_ACTIONS-1,colormap='Jet'), win='action_heat_map')
                current_state.step(action)

                reward = np.square(gt - previous_image) * 255 - np.square(gt - current_state.image) * 255
                reward = np.mean(reward, axis=1, keepdims=True)   # for label and state's channel > 1

                sum_reward += np.mean(reward) * np.power(opt.GAMMA, t)
            agent.stop_episode_and_train(current_state.image, reward, True)
            scheduler.step()
            vis.plot('total reward', sum_reward * 255)
            print("train total reward {a}".format(a=sum_reward * 255))
            vis.save([opt.name])

        # =========================Validation stage=========================

        for batch_idx, (state, targets, file_name) in enumerate(tqdm(VailLoader)):
            with torch.no_grad():
                l = targets.numpy()
                state = Kc_Glimpse.step_vail(state)
                current_state.reset(state.numpy())
                for t in range(opt.EPISODE_LEN):
                    action = agent.act(current_state.image)
                    current_state.step(action)

                mask = np.where(current_state.image != 0 , 1, 0)
                iou_val = iou_score(mask, targets.numpy())
                dice_val = dice(mask, targets.numpy())
                sens_val = sensitivity(mask, targets.numpy())
                ppv_val = ppv(mask, targets.numpy())
                hd95_val = hd95(mask, targets.numpy())
                BIOU_val = BIOU(mask, targets.numpy())
                # mask = mask[0].transpose(1, 2, 0) * 255
                # mask = np.squeeze(mask)
                # gt = np.asanyarray(l[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
                sum_all = Sum_vail(dice_val,iou_val,ppv_val,sens_val,hd95_val,BIOU_val)

        Mean_vail(opt, sum_all, VailLoader)


if __name__ == '__main__':
    opt = parser.parse_args()
    vis = Visualizer.Visualizer(opt.name)
    main(opt, vis)
