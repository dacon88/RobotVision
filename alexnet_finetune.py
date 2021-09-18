"""
alexnet_finetune helps setting up the alexnet model to finetune different classes .
Code takes inspiration from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from PIL import Image


class AlexnetFinetune:

    def __init__(self):
        # Top level data directory. Here we assume the format of the directory conforms
        #   to the ImageFolder structure
        self.data_dir = "/Users/davide/Documents/laurea_magistrale/second_semester_first_year/machine_learning/ML_project/iCubWorld1.0/human"

        # Number of classes in the dataset
        self.num_classes = 7

        # Batch size for training (change depending on how much memory you have)
        self.batch_size = 8

        # Number of epochs to train for
        self.num_epochs = 15

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = False
        # TODO: inspect what input size does. it should be the size of the data
        self.input_size = 128

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

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        self.dataloaders_dict = {
        x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = torchvision.models.alexnet(pretrained=False, num_classes=self.num_classes)

        # Send the model to GPU
        self.model = self.model.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        self.params_to_update = self.model.parameters()
        print("Params to learn:")
        if self.feature_extract:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        #TODO: take a look at the finetune parameters
        # Observe that all parameters are being optimized
        self.optimizer_ft = optim.SGD(self.params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        self.criterion = nn.CrossEntropyLoss()

        self.val_acc_history = []

    def train_model(self, is_inception=False):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
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
                for inputs, labels in self.dataloaders_dict[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer_ft.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    self.val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def predict_image(self, image_path):
        print("prediciton in progress")
        image = Image.open(image_path)

        transformation = self.data_transforms["val"]

        image_tensor = transformation(image)
        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.to('cuda')
            self.model.to('cuda')

        classes = self.dataloaders_dict['train'].dataset.classes
        print(classes)

        with torch.no_grad():
            output = self.model(image_tensor)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        probabilities = probabilities.numpy()

        index = probabilities.argmax()
        message = classes[index] + " " + str(probabilities[index]*100.) + " %"
        print(message)

        return index
