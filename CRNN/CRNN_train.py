import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
# from torchsummary import summary
# from imgaug import augmenters as iaa

# set path
data_path = "./dataset/"    # define UCF-101 RGB data path
action_name_path = './action.txt'
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
save_model_path = "./CRNN_ckpt/EfficientNet/20220422/"

# EncoderCNN architecture
CNN_fc_hidden1 = 1024
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
num_classes = 7             # number of target category 2
epochs = 36        # training epochs 60
batch_size = 120     # 100
learning_rate = 0.0001    # 5e-5

# Select which frame to begin & end in videos
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
begin_frame, end_frame, skip_frame = 5, 115, 5      # 5, 115, 5


def train(model, device, train_loader, optimizer):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    train_loss = []
    all_y = []
    all_y_pred = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)
        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        train_loss.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        # collect all y and y_pred in all batches
        all_y.extend(y)
        all_y_pred.extend(y_pred)

        loss.backward()
        optimizer.step()

        # show information
        tqdm.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}% \r'.format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))


    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    train_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    return sum(train_loss)/len(train_loss), train_score


def validation(model, device, test_loader, optimizer):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = []
    all_y = []
    all_y_pred = []
    N_count = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(test_loader)):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            N_count += X.size(0)

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y)
            test_loss.append(loss.item())                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            # show information
            tqdm.write('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}% \r'.format(
                epoch + 1, N_count, len(test_loader.dataset), 100. * (batch_idx + 1) / len(test_loader), loss.item(), 100 * step_score))


    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # save Pytorch models of best record
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return sum(test_loss)/len(test_loss), test_score

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True} if use_cuda else {}

## load UCF101 actions names
#with open(action_name_path, 'rb') as f:
#    action_names = pickle.load(f)
with open(action_name_path, 'r') as f:
    action_names = f.readline().split(",")

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)
#############################################################################
print(le.classes_)                                                          #
#############################################################################

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

# actions = []
# fnames = os.listdir(data_path)

# all_names = []
# for f in fnames:
#     loc1 = f.find('v_')
#     loc2 = f.find('_g')
#     actions.append(f[(loc1 + 2): loc2])

#     all_names.append(f)

# # print("all_names:", all_names)
# # print("actions:", actions)

actions = []
all_names = []

for action in os.listdir(data_path):
    for img_dir in os.listdir("{}/{}/image/".format(data_path, action)):
        actions.append(action)
        all_names.append(img_dir.split("/")[-1])

# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# # #########################################################################
# print(all_X_list)

# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(all_y_list)
# # #########################################################################

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)
# #########################################################################
# print(test_list)

# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(test_label)
# #########################################################################

transform_train = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ColorJitter(brightness=.05, hue=.05, saturation=.05),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(15),
                                transforms.RandomCrop([res_size, res_size]),
                                # iaa.Sequential([
                                #                 iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                                #             ]).augment_image,
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

transform_test = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, action_names, transform=transform_train), \
                       Dataset_CRNN(data_path, test_list, test_label, selected_frames, action_names, transform=transform_test)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Create model
# cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
cnn_encoder = EfficientNetCNNEncoder(fc_hidden1=CNN_fc_hidden1, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=num_classes).to(device)
# print model
print(cnn_encoder)
# summary(cnn_encoder, input_size = (batch_size, 110, 3, res_size, res_size))
print(rnn_decoder)
# summary(rnn_decoder, input_size = (batch_size, 110, CNN_embed_dim))

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.001)
warmup_scheduler = get_constant_schedule_with_warmup(optimizer, 5, last_epoch=-1)
lr_scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
writer = SummaryWriter("logs/EfficientNet/20220422/")
print("Creating logs")
print("Start Training")
# start training
for epoch in range(epochs):
    
    # train, test model
    train_losses, train_scores = train([cnn_encoder, rnn_decoder], device, train_loader, optimizer)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, valid_loader, optimizer)
    print("----------------------------------------------")
    if epoch < 5:
        warmup_scheduler.step()
    lr_scheduler.step()

    writer.add_scalar("Learning rate", get_lr(optimizer), epoch)
    writer.add_scalars("Loss", {"Train": train_losses}, epoch)
    writer.add_scalars("Loss", {"Test": epoch_test_loss}, epoch)
    writer.add_scalars("Accuracy", {"Train": train_scores}, epoch)
    writer.add_scalars("Accuracy", {"Test": epoch_test_score}, epoch)
    writer.flush()


    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    np.save('result/220422/CRNN_epoch_training_losses.npy', A)
    np.save('result/220422/CRNN_epoch_training_scores.npy', B)
    np.save('result/220422/CRNN_epoch_test_loss.npy', C)
    np.save('result/220422/CRNN_epoch_test_score.npy', D)

writer.close()


