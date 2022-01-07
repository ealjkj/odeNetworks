import numpy as np
from numpy.core.function_base import _logspace_dispatcher
from torch.cuda import device_of
from torch.optim import lr_scheduler
import copy
import time
import matplotlib.pyplot as plt
import torch
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(model, criterion, optimizer, scheduler, num_epochs, datasets_list, dataloaders_list, export_name=None):
    
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
    fig.tight_layout(pad=3.0)

    fig.suptitle('Accuracy an Loss')
    # ax1.plot(x, hist_loss_train, x, hist_loss_val)
    ax1.plot(x, hist_loss_train, label='train')
    ax1.plot(x, hist_loss_val, label='val')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    # ax2.plot(x, hist_acc_train, x, hist_acc_val)
    ax2.plot(x, hist_acc_train, label='train')
    ax2.plot(x, hist_acc_val, label='val')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    
    if export_name is not None:
        plt.savefig(export_name)
        
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
            
    return accuracy



def find_miss_classifications(model, loader, testset, label_names, export_name=None):
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

    if export_name is not None:
        plt.savefig(export_name)
    plt.show()


def apply_noise(img_arr, sigma = 0.1):
    mean = 0
    noise = torch.tensor(np.random.normal(mean, sigma, img_arr.shape), dtype=torch.float)
    return img_arr + noise

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_all_preds(model, loader):
    with torch.no_grad():
        all_preds = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
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
    return all_preds, all_labels

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()



def get_cm(model, loader, num_classes):
    #Matriz de confusión
    all_preds, all_labels = get_all_preds(model, loader)
    stacked = torch.stack(
        (
            all_labels
            ,all_preds
        )
        ,dim=1
    )

    cmt = torch.zeros(num_classes,num_classes, dtype=torch.int64)

    for p in stacked:
        tl, pl =p.tolist()
        tl, pl = int(tl), int(pl)
        cmt[tl, pl] = cmt[tl, pl] + 1

    return cmt.numpy()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, export_name=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if export_name is not None:
        plt.savefig(export_name)


def get_metrics(cmt):
    results = {}
    num_classes = cmt.shape[0]
    scores = np.zeros((num_classes, 3))
    test_counter = np.array([cmt[i,:].sum() for i in range(cmt.shape[0])])
    pre_acc = 0
    for i in range(num_classes):
        precision = cmt[i,i]/cmt[i,:].sum() 
        recall = cmt[i,i]/cmt[:,i].sum() if cmt[:,i].sum() != 0 else 0 
        f1score = 2*precision*recall/(precision + recall) if precision + recall != 0 else 0
        scores[i,:] = [precision, recall, f1score]
        pre_acc += cmt[i,i]

    results['accuracy'] = pre_acc/cmt.sum()
    results['macro precision'] = scores[:,0].sum()/num_classes
    results['macro recall'] = scores[:,1].sum()/num_classes
    results['macro f1score'] = scores[:,2].sum()/num_classes
    
    results['weighted precision'] = sum([test_counter[i]*scores[:,0][i] for i in range(4)])/test_counter.sum()
    results['weighted recall'] = sum([test_counter[i]*scores[:,1][i] for i in range(4)])/test_counter.sum()
    results['weighted f1score'] = sum([test_counter[i]*scores[:,2][i] for i in range(4)])/test_counter.sum()


    return results