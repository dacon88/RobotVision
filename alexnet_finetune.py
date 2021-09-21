"""
alexnet_finetune helps setting up the alexnet model to finetune different classes .
Code takes inspiration from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

import time
import os
import copy
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


class AlexnetFinetune:

    def __init__(self, img_size, mode="train"):

        # Number of classes in the dataset.
        # it corresponds to num of output of last layer
        self.num_classes = 10

        if mode == "train":
            self.model = torchvision.models.alexnet(pretrained=True) # , num_classes=self.num_classes)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        elif mode == "infer":
            self.model = torchvision.models.alexnet(num_classes=self.num_classes)

        self.fine_tuned = False

        self.input_size = int(img_size * 0.7)

        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.classes_names = ()

    def train_model(self, dataset_dir, batch_size, num_epochs):
        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), self.data_transforms[x]) for x in
                          ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                           num_workers=4) for x in ['train', 'val']}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to the device available
        self.model = self.model.to(device)

        # Gather the parameters to be optimized/updated in this run.
        params_to_update = self.model.parameters()
        # print("Params to learn:")
        for name, param in self.model.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

        # TODO: take a look at the finetune parameters
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        val_acc_history = []

        since = time.time()

        # best weights are now the pretrained ones
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders_dict[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients at every iteration
                    optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        #   In train mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()      # gradient of loss
                            optimizer_ft.step()  # update weights

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
            # stop training loop

            print()
        # stop epochs

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        self.classes_names = dataloaders_dict['train'].dataset.classes
        self.export_classes(file_name="classes_names.csv")

        self.fine_tuned = True

    def export_classes(self, file_name):
        with open(file_name, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.classes_names)
            print("created {0} containing class labels".format(file_name))

    def get_classes_names_from_csv(self, filename):
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            self.classes_names = list(reader)

    def save_nn_state(self, state_path):
        torch.save(self.model.state_dict(), state_path)
        print("Network weights saved in: {0}".format(state_path))

    def predict_image(self, image):

        start_time = time.time()

        transformation = self.data_transforms["val"]

        image_tensor = transformation(image)
        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.to('cuda')
            self.model.to('cuda')

        #print(classes)

        with torch.no_grad():
            output = self.model(image_tensor)
        # print(output[0])
        # output is sent to cpu for statistic analysis
        output = output.cpu()
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        probabilities = probabilities.numpy()

        # index has highest probability
        index = probabilities.argmax()
        prob_percentage = round((probabilities[index]*100.), 4)
        message = self.classes_names[0][index] + " " + str(prob_percentage) + " %"
        #print(message)

        # print("--- %s seconds ---" % (time.time() - start_time))

        return message
