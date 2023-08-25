
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import sys
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.losses import DiceLoss


# Reading the data from the directories
DATA_DIR = r"C:\Users\felix\Desktop\data"
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_label')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_label')


# Function to visualize some of the data when needed
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class CarlaLanesDataset(Dataset):
    CLASSES = ['background', 'left_marker', 'right_marker']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        get_label_name = lambda fn: re.sub(".png", "_label.png", fn)
        self.masks_fps = [os.path.join(masks_dir, get_label_name(image_id)) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            print(f"Error loading image: {self.images_fps[i]}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)  # Placeholder image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)
        if mask is None:
            print(f"Error loading mask: {self.masks_fps[i]}")
            mask = np.zeros((224, 224), dtype=np.uint8)  # Placeholder mask

        mask = mask.astype(np.uint8)
        new_width = (image.shape[1] // 32) * 32  # Calculate new width divisible by 32
        new_height = (image.shape[0] // 32) * 32  # Calculate new height divisible by 32
        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask = mask.astype(np.uint8)  # Convert mask to numpy array

        return image, mask

    def __len__(self):
        return len(self.ids)


dataset = CarlaLanesDataset(x_train_dir, y_train_dir, classes=CarlaLanesDataset.CLASSES)
image, mask = dataset[5] # get some image at a particular index

# Visualizing some image from the pytorch dataloader uncomment to apply
visualize(
    image=image,
    label = mask
)





def get_training_augmentation():
    train_transform = [
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0., shift_limit=0.1, p=1, border_mode=cv2.BORDER_CONSTANT),
        albu.GaussNoise(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.6,
        ),
    ]
    return albu.Compose(train_transform)



def get_validation_augmentation():
    return None

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn, mask=None),  # Apply preprocessing to the image only
        albu.Lambda(image=to_tensor, mask=None),
    ]
    return albu.Compose(_transform)



#  Visualize resulted augmented images and masks
augmented_dataset = CarlaLanesDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=CarlaLanesDataset.CLASSES,
)

#showing some of the images after applying the transformations on them
#Uncomment to visualize

for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(image=image, label=mask)




loss_string = 'multi_dice_loss'
ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'
DEVICE = 'cpu'

# Create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CarlaLanesDataset.CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = CarlaLanesDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CarlaLanesDataset.CLASSES,
)

valid_dataset = CarlaLanesDataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CarlaLanesDataset.CLASSES,
)

bs_train = 8
bs_valid = 8
train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=bs_valid, shuffle=False)
label_left = CarlaLanesDataset.CLASSES.index('left_marker')
label_right = CarlaLanesDataset.CLASSES.index('right_marker')

class MultiDiceLoss(base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BinaryDiceLossLeft = DiceLoss()
        self.BinaryDiceLossRight = DiceLoss()

    def forward(self, y_pr, y_gt):
        sigmoid = torch.nn.Sigmoid()
        y_pr = sigmoid(y_pr)
        left_gt = (y_gt == label_left)
        right_gt = (y_gt == label_right)
        loss_left = self.BinaryDiceLossLeft.forward(y_pr[:, label_left, :, :], left_gt)
        loss_right = self.BinaryDiceLossRight.forward(y_pr[:, label_right, :, :], right_gt)
        return (loss_left + loss_right) * 0.5

metrics = []
loss = MultiDiceLoss()
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=1e-4),
])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# Train model
epochs = 100
best_loss = 1e10
for i in range(epochs):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # Do something (save model, change lr, etc.)
    if best_loss > valid_logs[loss_string]:
        best_loss = valid_logs[loss_string]
        torch.save(model, './best_model_{}.pth'.format(loss_string))
        print('Model saved!')

    if i == 3:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
