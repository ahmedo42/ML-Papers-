import argparse
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from GoogLeNet import GoogLeNet

parser = argparse.ArgumentParser(
    description="configure training hyperparameters")
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--aux', action='store_false',
                    help='auxillary classifiers')

args = parser.parse_args()
writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate(val_loader, model, criterion):

    model.eval()
    total = correct = loss = 0
    with torch.no_grad():
      for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs,aux1,aux2 = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted.cpu() == labels.cpu()).sum()
          loss += criterion(outputs,labels)

    acc = 100 * correct/total
    return loss , acc


def train(train_loader, model,criterion,optimizer,scheduler,epochs):
    best_acc = 0
    for epoch in range(epochs):
        correct = 0
        total = 0
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output,aux1,aux2 = model(images)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            aux1_loss = aux2_loss = 0
            if args.aux:
                aux1_loss = 0.3*criterion(aux1,labels) 
                aux2_loss = 0.3*criterion(aux2,labels) 

            loss +=  aux1_loss + aux2_loss
            loss.backward()
            train_loss += aux1_loss + aux2_loss + loss
            optimizer.step()
        train_acc = 100* correct/total
        val_loss , val_acc = validate(test_loader, model, criterion)
        writer.add_scalar('Loss/Train',train_loss,epoch)
        writer.add_scalar('Accuracy/Train',train_acc,epoch)
        writer.add_scalar('Loss/Val',val_loss,epoch)
        writer.add_scalar('Accuracy/Val',val_acc,epoch)
        if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'googlenet_cifar10.pth')
        print("Epoch: {} ,Validation Accuracy: {:.2f} ,Train Accuracy: {:.2f}".format(epoch,val_acc,train_acc))
        model.train()
        scheduler.step(val_loss)


if __name__ == "__main__":
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_data = torchvision.datasets.CIFAR10(
        '../data/CIFAR10/train', transform=train_transform, train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(
        '../data/CIFAR10/test', transform=test_transform, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = GoogLeNet(n_classes=10,aux_logits=args.aux)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=5)
    model.cuda()
    train(train_loader,model,criterion,optimizer,scheduler,EPOCHS)
    writer.flush()
    print("Compeleted Training")
