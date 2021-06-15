import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from network2 import SoundClassifyNet2D
import os
from dataGenerator import to_device, get_dataset, DeviceDataLoader
import torch.optim as optim
import streamlit as st
# obtain the available device
def get_default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device=get_default_device()
#print(device)

# function to plot losses and accuracy over time
def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()

def plot_accuracy(history):
    losses = [x['val_acc'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()
def makedataloader(features,labels,device):
#    features = torch.load(fea)
#    labels = torch.load(label)
# features = torch.load("data/features_mfccs_2D.pt")
# labels = torch.load("data/labels_2D.pt")
    dataset = get_dataset(features, labels)

# split the training data into training and validation data
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

# Initialize the data loader
    batch_size = 32
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_ds, train_size, shuffle=True, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, val_size, shuffle=True, num_workers=0, pin_memory=True)

# moving the data to the device
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    return train_loader,val_loader

# evaluate function on validation set
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# train on mfcc data
def train(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, print_loss=True):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    best_acc = 1e-10
    n_save_epoch = 500
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 180], gamma=0.1)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1)
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        with torch.no_grad():
            result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        history.append(result)
        if print_loss:
            print(
                f"lr: [{optimizer.state_dict()['param_groups'][0]['lr']}],Epoch [{epoch}]: Validation loss -> {result['val_loss']}, Validation accuracy -> {result['val_acc']}")
            st.write("lr:",optimizer.state_dict()['param_groups'][0]['lr'],"Epoch:",epoch,"Validation loss ->",result['val_loss'],"Validation accuracy ->",result['val_acc'])
        if result['val_acc'] >= best_acc:
#            model_dir = '../model/'
#             save_model_path = os.path.join(model_dir, 'Sound_Classify_model2D_%d.pth' % epoch)
#            save_model_path = os.path.join(model_dir, 'Sound_Classify_model2D_0h.pth')
#            torch.save(model.state_dict(), save_model_path)
            best_acc = result['val_acc']

        scheduler.step()
    return history
'''
# input_size = 80
# input_size = 40
output_size = 2
# model = SoundClassifyNet(input_size, output_size)
model = SoundClassifyNet2D(output_size)
print(model)
to_device(model, device)
history = [evaluate(model, val_loader)]

epochs = 100
lr = 1e-2
history += train(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam)

plot_accuracy(history)
plot_losses(history)

model_dir = '../model/'
save_model_path = os.path.join(model_dir, 'Sound_Classify_model2D_0h_end.pth')
torch.save(model.state_dict(), save_model_path)
'''
def start(epochs,lr,optim,features,labels):
    device = get_default_device()
    train_loader,val_loader = makedataloader(features,labels,device)
    output_size = 2
    model = SoundClassifyNet2D(output_size)
    device = get_default_device()
    print(device)
    to_device(model,device)
    lr = lr
    if optim == "Adam":
        opt_func = torch.optim.Adam
    elif optim == "SGD":
        opt_func = torch.optim.SGD
    else:
        opt_func = torch.optim.RMSprop
    history = [evaluate(model, val_loader)]
    history += train(epochs, lr, model, train_loader, val_loader, opt_func)
    model_dir = '../model/'
    save_model_path = os.path.join(model_dir, 'streamlit-64-1.pth')
    torch.save(model.state_dict(), save_model_path)
    return model.state_dict()
#features = torch.load("features_mfcc_float.pt")
#labels = torch.load("labels.pt")
#start(10,0.01,"Adam",features,labels)
def what():
    torch.cuda.is_available()
'''
fea  = torch.load("../model/features_mfcc_float.pt")
lab = torch.load("../model/labels.pt")
start(100,0.01,'Adam',fea,lab)
'''