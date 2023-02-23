import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import shap
import time, sys

from ResNet import Bottleneck, ResNet, ResNet50, ResNet18, ResNet34, ResNet101, ResNet152
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import models
from torchsummary import summary
from datetime import datetime
from pytz import timezone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mdsm_width = 11
mdsm_height = 108

EPOCHS = 400
BATCH_SIZE = 512
net_type = "ResNet18"

path = "/tf/data/MDSM_ResNetbased/"
trans_stat = True

def mix_random(col, row, mdsm_body):
    size_suffle = random.randint(0,10)
    switchsource = torch.randint(0, row - 1, (size_suffle,))
    temp = np.zeros((1, col), np.float32)

    for i in range(0, int(size_suffle)):
        if i == switchsource[i]:
            continue
        temp = mdsm_body[i, :].copy()
        mdsm_body[i, :] = mdsm_body[switchsource[i], :].copy()
        mdsm_body[switchsource[i], :] = temp.copy()
    return torch.tensor(mdsm_body)

def flip_random(col, row, mdsm_body):
    size_suffle = random.randint(0,12)
    if size_suffle % 4 != 0:
        return torch.tensor(mdsm_body)

    int_row = int(row)
    for i in range(0, int(int_row / 2)):
        temp = mdsm_body[i, :].copy()
        mdsm_body[i, :] = mdsm_body[int_row - i - 1, :].copy()
        mdsm_body[int_row - i - 1, :] = temp.copy()
    return torch.tensor(mdsm_body)

class MDSMDataset(Dataset):
    def __init__(self, mdsmdata_file):
        self.df = pd.read_csv(mdsmdata_file)
        rating = self.df[['ReviewID', 'reviewStar']]
        self.rating = rating.drop_duplicates('ReviewID')
        if trans_stat == True:
            self.height = self.df['ReviewID'].value_counts().max()
        else:
            #Hardcoding for current dataset
            self.height = 108

        mdsm_body = self.df.drop(['reviewNo', 'reviewStar', 'mGNR'], axis=1)
        mdsm_body['imageCnt'] = (mdsm_body['imageCnt'] - mdsm_body['imageCnt'].min())/ (mdsm_body['imageCnt'].max() - mdsm_body['imageCnt'].min())
        mdsm_body['helpfulCnt'] = (mdsm_body['helpfulCnt'] - mdsm_body['helpfulCnt'].mean())/ mdsm_body['helpfulCnt'].std()
        body_height, body_width = mdsm_body.shape;
        self.width = body_width - 1
        mdsm_width = self.width
        mdsm_height = self.height

        dummy_mdsd = np.zeros((body_height, self.height, self.width), np.float32)
        mdsm_index = np.zeros(self.rating['ReviewID'].max()+1, int)
        mdsm_count = np.zeros(self.rating['ReviewID'].max()+1, int)
        mdsm_index.fill(-1)

        max_index = int(0)
        for index, body in mdsm_body.iterrows():
            dummy_index = max_index
            if mdsm_index[int(body['ReviewID'])] != -1:
                dummy_index = mdsm_index[int(body['ReviewID'])]
            else:
                mdsm_index[int(body['ReviewID'])] = dummy_index
                max_index = max_index + 1

            dummy_mdsd[dummy_index, mdsm_count[dummy_index]] = body.drop('ReviewID')
            mdsm_count[dummy_index] = mdsm_count[dummy_index] + 1

        self.mdsm_body = dummy_mdsd

    def __len__(self):
        return self.rating.shape[0]


    def __getitem__(self, idx):
        if trans_stat == True:
            _tensor = flip_random(self.width, self.height, self.mdsm_body[idx])
        else:
            _tensor = torch.tensor(self.mdsm_body[idx])
        rtn_tensor = _tensor.unsqueeze(0)
        return rtn_tensor, self.rating.iloc[idx, 1]


if net_type == "ResNet18":
    net = ResNet18(6, 1).to('cuda')
    print("ResNet18 is used")
elif net_type == "ResNet34":
    net = ResNet34(6, 1).to('cuda')
    print("ResNet34 is used")
elif net_type == "ResNet50":
    net = ResNet50(6, 1).to('cuda')
    print("ResNet50 is used")
elif net_type == "ResNet101":
    net = ResNet101(6, 1).to('cuda')
    print("ResNet101 is used")
elif net_type == "ResNet152":
    net = ResNet152(6, 1).to('cuda')
    print("ResNet152 is used")

summary(net, (1, mdsm_height, mdsm_width))

print('-- Loading dataset--')

dataset = MDSMDataset(path+'amazon_hmdvr_df_tokenized_sentiment_score_extended_normalized.csv')
train_size = round(len(dataset) * 0.8)
test_size = len(dataset) - train_size

print("Train(", train_size, ") vs Test(", test_size, ")")

print('-- Building train and test dataset / dataloader--')
train_dataset, test_dataset = random_split(dataset, [int(train_size),int(test_size)])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)

classes = ['0', '1', '2', '3', '4', '5']

def calcu_metric(outputs, labels):
    mae = abs(outputs - labels)
    mse = torch.sqrt(mae)
    return mae, mse

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

correct = 0
total = 0


print('-- Start training : ', EPOCHS, 'epochs')
start = time.time()

mse_history = {'train': [], 'val': []}
mae_history = {'train': [], 'val': []}
loss_history = {'train': [], 'val': []}
acc_history = {'train': [], 'val': []}

for epoch in range(EPOCHS):
    losses = []

    mse = np.zeros(train_size, np.float32)
    mae = np.zeros(train_size, np.float32)


    val_mse = np.zeros(test_size, np.float32)
    val_mae = np.zeros(test_size, np.float32)

    metric_index = 0

    running_loss = 0
    train_loss = 0
    train_acc = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss += loss.item()

        pred = outputs.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(labels.data.view_as(pred)).sum()


        _mae, _mse = calcu_metric(pred.squeeze(), labels)
        #print(_mse.detach().cpu().numpy())
        mae[metric_index:metric_index+len(inputs)] = _mae.detach().cpu().numpy()
        mse[metric_index:metric_index+len(inputs)] = _mse.detach().cpu().numpy()
        #print(torch.tensor(mse))
        metric_index += len(inputs)

        if i%100 == 0 and i > 0:
            print(f'Loss [{epoch+1}/{EPOCHS}, {i}](epoch, minibatch): ', f'{running_loss / 100:.5f}')
            running_loss = 0.0

    mae_epoch = torch.mean(torch.tensor(mae))
    mse_epoch = torch.mean(torch.tensor(mse))

    mse_history['train'].append(mse_epoch)
    mae_history['train'].append(mae_epoch)
    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)

    metric_index = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)

            pred = outputs.data.max(1, keepdim=True)[1]
            _mae, _mse = calcu_metric(pred.squeeze(), labels)
            val_mae[metric_index:metric_index+len(images)] = _mae.detach().cpu().numpy()
            val_mse[metric_index:metric_index+len(images)] = _mse.detach().cpu().numpy()
            metric_index += len(images)

    val_mae_epoch = torch.mean(torch.tensor(val_mae))
    val_mse_epoch = torch.mean(torch.tensor(val_mse))
    mse_history['val'].append(val_mse_epoch)
    mae_history['val'].append(val_mae_epoch)
    acc_history['train'].append((100. * train_acc / len(trainloader.dataset)).detach().cpu().numpy())

    train_loss /= len(trainloader.dataset)
    if EPOCHS > 50:
        if epoch % 5 == 0:
            print('Epoch: {}/{} Avg. loss:{:.4f} Acc.: {:.4f}% '.format(epoch, EPOCHS, train_loss, 100. * train_acc / len(trainloader.dataset)), f"MAE : {mae_epoch.item():.3f}", f"MSE : {mse_epoch.item():.3f}", f"VAL_MAE : {val_mae_epoch.item():.3f}", f"VAL_MSE : {val_mse_epoch.item():.3f}" ")")
    else:
        print('Epoch: {}/{} Avg. loss:{:.4f} Acc.: {:.4f}% '.format(epoch, EPOCHS, train_loss, 100. * train_acc / len(trainloader.dataset)), f"MAE : {mae_epoch.item():.3f}", f"MSE : {mse_epoch.item():.3f}", f"VAL_MAE : {val_mae_epoch.item():.3f}", f"VAL_MSE : {val_mse_epoch.item():.3f}" ")")

print('Training Done')
trans_stat = False
end = time.time()
print(f"{net_type} training takes {end - start:.5f} sec")

now = datetime.now(timezone('Asia/Seoul'))

hist_csv = np.stack([mae_history['val'], mse_history['val'], mae_history['train'],
                     mse_history['train'], acc_history['train']], 1)
hist_csv_df = pd.DataFrame(hist_csv)
hist_csv_df.columns = ['validation_mae', 'validation_mse', 'train_mae', 'train_mse', 'train_accuracy']
time_str = now.strftime('%Y-%m-%d_%H:%M:%S')
hist_csv_df.to_csv(path+"training_metrics/amazon_hmdvr_df_tokenized_sentiment_score_model-{}_epochs-{}-batch-{}-{}.csv"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str), index=False)
print(path+"training_metrics/amazon_hmdvr_df_tokenized_sentiment_score_model-{}_epochs-{}-batch-{}-{}.csv saved"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str))

torch.save({
                "epoch": EPOCHS,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
           "{}check_points/amazon_hmdvr_df_tokenized_sentiment_score_model-{}_epochs-{}-batch-{}-{}.pt"
                   .format(path,net_type, EPOCHS, BATCH_SIZE, time_str))
torch.save({
                "epoch": EPOCHS,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },path+"check_points/latest.pt")
torch.save(net, "{}check_points/amazon_hmdvr_df_tokenized_sentiment_score_model-{}_epochs-{}-batch-{}-{}.model"
                   .format(path,net_type, EPOCHS, BATCH_SIZE, time_str))
torch.save(net, path+"check_points/latest.model")

print(path+"check_points/amazon_hmdvr_df_tokenized_sentiment_score_model-{}_epochs-{}-batch-{}-{}.pt & check_points/latest.pt saved"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str))

plt.title("Mean Abs Error [STR] : model-{}_epochs-{}_batch-{}-{}"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str))
plt.plot(range(1,EPOCHS+1),mae_history["train"],label="train_mae")
plt.plot(range(1,EPOCHS+1),mae_history["val"],label="validation_mae")
plt.ylabel("Mean Abs Error [STR]")
plt.xlabel("Training Epochs")
plt.ylim([0,1.5])
plt.legend()
plt.savefig(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_mae_model-{}_epochs-{}-batch-{}-{}.png"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)
plt.clf()

plt.title("Mean Square Error [$STR^2$] : model-{}_epochs-{}_batch-{}-{}"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str))
plt.plot(range(1,EPOCHS+1),mse_history["train"],label="train_mse")
plt.plot(range(1,EPOCHS+1),mse_history["val"],label="validation_mse")
plt.ylabel("Mean Square Error [$STR^2$]")
plt.xlabel("Training Epochs")
plt.ylim([0,1.5])
plt.legend()
plt.savefig(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_mse_model-{}_epochs-{}-batch-{}-{}.png"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)

plt.clf()
plt.title("Train Accuracy : model-{}_epochs-{}_batch-{}-{}"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str))
plt.plot(range(1,EPOCHS+1),acc_history["train"],label="train_accuracy")
#plt.plot(range(1,EPOCHS+1),mse_history["val"],label="validation mse")
plt.ylabel("Train Accuracy")
plt.xlabel("Training Epochs")
#plt.ylim([0,1.5])
plt.legend()
plt.savefig(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_acc_model-{}_epochs-{}-batch-{}-{}.png"
                   .format(net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)

batch = next(iter(testloader))
images, _ = batch

#max_size = BATCH_SIZE - 3
max_size = 100
shap_test_size = max_size + 3

background = images[:max_size]
test_images = images[max_size:shap_test_size]

#e = shap.DeepExplainer(net, background.to(device))
e = shap.DeepExplainer(net, background.to(device))
#shap_values = e.shap_values(test_images)

df = pd.read_csv(path+'amazon_hmdvr_df_tokenized_sentiment_score_extended_normalized.csv')
dff = df.drop(['reviewNo', 'ReviewID', 'reviewStar', 'mGNR'], axis=1)

shap.initjs()

print('-- Building shap test dataset / dataloader--')
for i in range(5):
    dataset_shap = MDSMDataset(path+f'amazon_hmdvr_df_tokenized_sentiment_score_extended_normalized_reviewStar{i+1}.csv')
    print(path+f'amazon_hmdvr_df_tokenized_sentiment_score_extended_normalized_reviewStar{i+1}.csv is used for shap analysis')

    shap_shap_loader = torch.utils.data.DataLoader(dataset_shap, batch_size = len(dataset_shap), shuffle=True, num_workers=0)

    batch_shap = next(iter(shap_shap_loader))
    images_shap, _ = batch_shap

    max_size_shap = len(dataset_shap)
  
    test_images_shap = images_shap[:max_size_shap]
    shap_values_shap = e.shap_values(test_images_shap)
    
    print(f'amazon_hmdvr_df_tokenized_sentiment_score_extended_normalized_reviewStar{i+1}.csv is used for shap analysis')
    
    
    
    plt.clf()
    shap.summary_plot(shap_values_shap[0][0][0], images_shap[:][0][0], feature_names=dff.columns,show=False)
    plt.savefig(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_shap_summary-data_{}-{}_model-{}_epochs-{}-batch-{}-{}.png"
            .format(i+1, max_size, net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)
    print(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_shap_summary-data_{}-{}_model-{}_epochs-{}-batch-{}-{}.png is saved"
            .format(i+1, max_size, net_type, EPOCHS, BATCH_SIZE, time_str))
    
    plt.clf()
    shap.summary_plot(shap_values_shap[0][0][0], images_shap[:][0][0], feature_names=dff.columns,plot_type='bar',show=False)
    plt.savefig(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_shap_summary-bar-data_{}-{}_model-{}_epochs-{}-batch-{}-{}.png"
            .format(i+1, max_size, net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)
    print(path+ "graphs/amazon_hmdvr_df_tokenized_sentiment_score_shap_summary-bar-data_{}-{}_model-{}_epochs-{}-batch-{}-{}.png is saved"
            .format(i+1, max_size, net_type, EPOCHS, BATCH_SIZE, time_str))
# plt.clf()
# shap.summary_plot(shap_values[0][0][0], images[:][0][0], feature_names=dff.columns,show=False)
# plt.savefig("graphs/amazon_hmdvr_df_tokenized_sentiment_score_shap-{}_summary_epochs-{}-batch-{}-{}.png"
#                    .format(max_size, net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)
# plt.clf()
# shap.summary_plot(shap_values[0][0][0], images[:][0][0], feature_names=dff.columns,plot_type='bar',show=False)
# plt.savefig("graphs/amazon_hmdvr_df_tokenized_sentiment_score_shap_summary_bar-{}_model-{}_epochs-{}-batch-{}-{}.png"
#                    .format(max_size, net_type, EPOCHS, BATCH_SIZE, time_str), dpi=300)