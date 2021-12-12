import numpy as np
from torch.optim import lr_scheduler
import copy
import time
import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, datasets_list, dataloaders_list):
    
    train_dataset = datasets_list[0]
    test_dataset = datasets_list[1]

    train_dataloader = dataloaders_list[0]
    test_dataloader = dataloaders_list[1]

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

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/content/gdrive/MyDrive/DATASETS/Polen/outputs/model.pth')

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

    return model

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

dataloaders = {'train': train_loader, 'val': test_loader}
dataset_sizes  = {'train': len(train_dataset), 'val': len(test_dataset)}

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=num_epochs)


                                      