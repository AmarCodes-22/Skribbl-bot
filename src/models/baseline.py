import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms


class BaseLine:
    def __init__(self) -> None:
        self.input_image_size = 224
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, model=None, criterion=None, optimizer=None, num_epochs=2):
        """Training loop"""
        if model is None:
            self.model = models.resnet18(pretrained=True)
        else:
            self.model = model

        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        since = time.time()

        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.optimizer.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(self.model.state_dict())

                print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:0.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        self.model.load_state_dict(best_model_weights)
        return self.model

    def create_dataloaders(
        self,
        datasets_dict,
        dataset_transforms = None,
        batch_size: int = 4,
        shuffle: bool = True,
        **kwargs
    ):
        """Using datasets_dict that we get from quickdraw-version, create dataloaders"""
        self.dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'test']}
        self.class_names = datasets_dict["train"].classes

        if dataset_transforms is None:
            self.transform = self._load_transforms()
        else:
            self.transform = dataset_transforms

        datasets_dict["train"].transform = self.transform["train"]
        datasets_dict["test"].transform = self.transform["train"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                datasets_dict[x], batch_size=batch_size, shuffle=shuffle, **kwargs
            )
            for x in ["train", "test"]
        }

    def visualize_model(self, model, num_images=6):
        """Visualize some of the model outputs"""
        assert self.dataloaders is None

        was_training = model.training
        model.eval()
        images_drawn = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders["test"]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_drawn += 1
                    ax = plt.subplot(num_images // 2, 2, images_drawn)
                    ax.axis("off")
                    ax.set_title(
                        "predicted: {}, correct: {}".format(
                            self.class_names[preds[j]], "meh"
                        )
                    )
                    self._imshow(inputs.cpu().data[j])

                    if images_drawn == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def _load_transforms(self):
        """Load basic transforms"""
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(self.input_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(self.input_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        return data_transforms

    def _imshow(self, inp, title=None):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        inp = inp.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        plt.figure(figsize=(12,3))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
