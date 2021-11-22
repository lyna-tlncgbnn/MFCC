import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import numpy as np

# calculates the accuracy of predictions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# print precision-recall curve
def calculate_PR(labels, preds, pred_val):
    preds_np = preds.detach().cpu().numpy()
    preds_val_np = pred_val.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    print('roc_auc_score: ', roc_auc_score(labels_np, preds_np))
    cm = confusion_matrix(labels_np, preds_np)
    print('confusion_matrix: ')
    print(cm)
    precision = precision_score(labels_np, preds_np)
    print('precision: ', precision)
    recall = recall_score(labels_np, preds_np)
    print('recall: ', recall)
    f1_sc = f1_score(labels_np, preds_np)
    print('f1_score: ', f1_sc)

def draw_ROC(labels, preds):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels_np, preds_np, pos_label=1)
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('ROC of test examples')
    plt.show()

def draw_PR(labels, preds):
    # Precision-recall curve
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    # preds_np = 1 - preds_np
    # labels_np = 1 - labels_np
    precision, recall, thresholds = precision_recall_curve(labels_np, preds_np)
    precision = np.fliplr([precision])[0]
    recall = np.fliplr([recall])[0]
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig(output_folder + "Precision_recall.png")

# def test_accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     print('preds[0]: ', preds + preds)
#     print('preds: ', preds.shape)
#     # print('gt: ', labels)
#     res = (preds == labels).squeeze()
#     class_correct = list(0. for i in range(2))
#     class_total = list(0. for i in range(2))
#     for idx in range(len(labels)):
#         label = labels[idx]
#         class_correct[label] += res[idx].item()
#         class_total[label] += 1
#     # print(class_correct)
#     preds_1 = outputs[:, 1]
#     calculate_PR(labels, preds, preds_1)
#     draw_ROC(labels, preds_1)
#     draw_PR(labels, preds_1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds)), class_correct, class_total

def pred(outputs1, outputs2, outputs3):
    _, preds1 = torch.max(outputs1, dim=1)
    _, preds2 = torch.max(outputs2, dim=1)
    _, preds3 = torch.max(outputs3, dim=1)
    pred = (preds1 + preds2 + preds3) / 3
    for i in len(pred):
        if pred[i] >= 1 - pred[i]:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred
            

class AudioClassificationBase(nn.Module):
    """
    definition of the base model
    """
    def training_step(self, batch):
        images, labels = batch
        # print(images.shape)
        # generate output
        _,out = self(images)
        # calculate loss
        weights = torch.Tensor([0.5, 1]).cuda()
        loss = F.cross_entropy(out, labels, weight=weights)
        # loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        _,out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        # combine accuracy
        epoch_accs = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_accs}

    def test_step(self, batch):
        classes = ('bad', 'good')
        images, labels = batch
        _,out = self(images)
#         print(out)
#         print(out.shape)
        acc, class_correct, class_total, pred = test_accuracy(out, labels)
        for i in range(2):
            print('Accuracy of %5s : %2f' % (classes[i],  class_correct[i] / class_total[i]))
        return {'test_acc':acc}, pred[0]

    def test_epoch_end(self, outputs):
        batch_accs = [x['test_acc'] for x in outputs]
        epoch_accs = torch.stack(batch_accs).mean()
        print('Accuracy of all test examples: %2f' % epoch_accs)
        return {'test_acc': epoch_accs}
    
def test_accuracy(outputs1, labels):
    _, preds1 = torch.max(outputs1, dim=1)
    print(preds1)
#     _, preds2 = torch.max(outputs2, dim=1)
#     _, preds3 = torch.max(outputs3, dim=1)
#     preds = (preds1 + preds2 + preds3)
#     for i in range(len(preds)):
#         if preds[i] <= 1:
#             preds[i] = 0
#         else:
#             preds[i] = 1
            
#     print('preds1', preds1)
#     print('preds2', preds2)
#     print('preds3', preds3)
#     print('preds ', preds)
#     print('preds[0]: ', preds + preds)
#     print('preds: ', preds.shape)
    # print('gt: ', labels)
    res = (preds1 == labels).squeeze()
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    for idx in range(len(labels)):
        label = labels[idx]
        class_correct[label] += res[idx].item()
        class_total[label] += 1

        
    print('test_accuracy has been used')
    # print(class_correct)
#     preds_1 = outputs[:, 1]
#     calculate_PR(labels, preds, preds_1)
#     draw_ROC(labels, preds_1)
#     draw_PR(labels, preds_1)
    return torch.tensor(torch.sum(preds1 == labels).item() / len(preds1)), class_correct, class_total, preds1
    
def test_epoch_end(outputs):
    batch_accs = [x['test_acc'] for x in outputs]
    epoch_accs = torch.stack(batch_accs).mean()
    print('Accuracy of all test examples: %2f' % epoch_accs)
    print('test_epoch_end has been used')
    return {'test_acc': epoch_accs}
    
def test_step(model1, model2, model3, batch1, batch2, batch3):
    classes = ('bad', 'good')
    images1, labels = batch1
    images2, labels = batch2
    images3, labels = batch3
    out1 = model1(images1)
    out2 = model2(images2)
    out3 = model3(images3)
    print('images1.shape:', images1.shape)
    print('out1.shape:',out1.shape)
    acc, class_correct, class_total, _ = test_accuracy(out1, out2, out3, labels)
    print('test_step has been used')
    for i in range(2):
        print('Accuracy of %5s : %2f' % (classes[i],  class_correct[i] / class_total[i]))
    return {'test_acc': acc}
    
    
class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResModule, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01),
#             nn.Dropout(0.4),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels)
        )
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv_block(x)
        x1 = self.conv(x)
#         print('x shape:', x.shape)
#         print('out shape', out.shape)
#         print('x1 shape:', x1.shape)
        
        out += x1
        out = self.BatchNorm(out)
        out = self.activation(out)
#         print('final out', out.shape)
#         print('delimiter')
        return out

class SoundClassifyNet2D(AudioClassificationBase):
    def __init__(self, output_class):
        """
        classification model using Convolution
            :param input_size: size of the input data, which means the dimensions of MFCCs feature here
            :param output_size: the number of classes, set to 2 for binary classification
        """
        super(SoundClassifyNet2D, self).__init__()

        data_channels = 1
        self.ResNet = nn.Sequential(
            ResModule(16, 32),
            ResModule(32, 64),
#             ResModule(32, 64),
            ResModule(64, 128),
#             ResModule(128, 256),
#             ResModule(256, 512),
            
#             ResModule(128, 256),
            # ResModule(128, 128),
            # ResModule(256, 512),

            nn.AdaptiveMaxPool2d((20, 128))
#             nn.AdaptiveMaxPool2d((20, 256))
        )

        self.ConvBlock = nn.Sequential(
#             nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
#             nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.fc_layer = nn.Sequential(
            
#             nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Linear(16, output_class),
            nn.Softmax(dim=1)
        )
        
        self.lstm = nn.LSTM(128, 256, 4)
        
        self.tail = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
        )
        self.end = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 3),
            nn.Linear(3,output_class),
            nn.Softmax(dim=1)
        )
        self.start = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x_batch):
        out = self.start(x_batch)
        out = self.ResNet(out)
        out = torch.mean(out, dim=2)
        out = torch.mean(out, dim=1)
#         out = self.ConvBlock(out)
        out = torch.squeeze(out)
#         out = out.unsqueeze(dim=0)
#         out, _ = self.lstm(out)
#         out = out.squeeze(dim=0)
        # Add LSTM here
#         out = self.fc_layer(out)
#         print(out.shape)
        res = self.tail(out)
        out = self.end(res)
        return res,out


# model = SoundClassifyNet2D(2)
# data = torch.randn(64, 1, 20, 300)
# output = model(data)
# print(output.shape)