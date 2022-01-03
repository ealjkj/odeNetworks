import numpy as np
from numpy.core.function_base import _logspace_dispatcher
from torch.cuda import device_of
from torch.optim import lr_scheduler
import copy
import time
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(model, criterion, optimizer, scheduler, num_epochs, datasets_list, dataloaders_list):
    
    train_dataset = datasets_list[0]
    test_dataset = datasets_list[1]

    train_loader = dataloaders_list[0]
    test_loader = dataloaders_list[1]

    dataloaders = {'train': train_loader, 'val': test_loader}
    dataset_sizes  = {'train': len(train_dataset), 'val': len(test_dataset)}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    hist_loss_train = np.zeros(num_epochs)
    hist_loss_val = np.zeros(num_epochs)
    hist_acc_train = np.zeros(num_epochs)
    hist_acc_val = np.zeros(num_epochs)
    x = np.array(range(num_epochs))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                hist_loss_train[epoch] = epoch_loss
                hist_acc_train[epoch] = epoch_acc
            else:
                hist_loss_val[epoch] = epoch_loss
                hist_acc_val[epoch] = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #         }, '/content/gdrive/MyDrive/DATASETS/Polen/outputs/model.pth')

    # #Plot acc and val
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(x, hist_loss_train)
    # axs[0, 0].set_title('train loss')
    # axs[0, 1].plot(x, hist_loss_val, 'tab:orange')
    # axs[0, 1].set_title('validation loss')
    # axs[1, 0].plot(x, hist_acc_train, 'tab:green')
    # axs[1, 0].set_title('train accuracy')
    # axs[1, 1].plot(x, hist_acc_val, 'tab:red')
    # axs[1, 1].set_title('validation accuracy')

    
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # ploting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Accuracy an Loss')
    ax1.plot(x, hist_loss_train, x, hist_loss_val)
    ax2.plot(x, hist_acc_train, x, hist_acc_val)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.show()


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def results(model, loader, num_classes):
    with torch.no_grad():
        correct = 0
        total = 0
        test_counter = [0]*num_classes
        all_preds = torch.tensor([]).cuda()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            all_preds = torch.cat((all_preds, outputs),dim=0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for idx in range(num_classes):
                test_counter[idx] += (labels == idx).sum().item()
        
        accuracy = 100 * correct / total
            
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print(test_counter)



def find_miss_classifications(model, loader, testset, label_names):
    with torch.no_grad():
        all_preds = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
        counter = 0
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds = torch.cat(
                (all_preds, predicted)
                ,dim=0
            )

            all_labels = torch.cat(
                (all_labels, labels)
                ,dim=0
            )

        v = ~all_preds.eq(all_labels)
        indices = torch.tensor(range(len(v)))[v]
        all_preds = all_preds[v]
        all_labels = all_labels[v]

    f, axarr = plt.subplots(3,3)
    f.set_size_inches(8, 8)
    f.tight_layout(pad=3.0)

    for i in range(3):
        for j in range(3):
            idx = indices[i*3 + j]
            img = testset[idx.item()][0].permute(1, 2, 0)
            correct_label = testset[idx.item()][1]
            incorrect_prediction = int(all_preds[idx.item()].item())
            axarr[i,j].imshow(img)   
            axarr[i,j].set_xlabel(label_names[correct_label], fontsize=16, color = 'green', fontfamily="sans-serif")
            axarr[i,j].set_ylabel(label_names[incorrect_prediction], fontsize = 16, color = 'red')
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])

    plt.show()

