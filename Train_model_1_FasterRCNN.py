
import os
import json
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class CustomDataset(Dataset):
    def __init__(self, json_file, image_folder, transform=None):
        self.transform = transform
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.image_folder = image_folder
        self.images = data["images"]
        self.annotations = data["annotations"]

        self.image_to_annotations = {}
        for annotation in self.annotations:
            image_id = annotation["image_id"]
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(annotation)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_info = self.images[annotation["image_id"]]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.image_folder, file_name)

        image = Image.open(image_path).convert("RGB")

        boxes = self.convert_bbox(annotation["bbox"])
        labels = annotation["category_id"]

        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([labels], dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}

    def convert_bbox(self, bbox):
        x, y, width, height = bbox
        return [x, y, x + width, y + height]


train_json_file = '/kaggle/input/cv-ass-4-dataset/annotations/instances_train2017.json'
train_image = '/kaggle/input/cv-ass-4-dataset/train2017'

val_json_file = '/kaggle/input/cv-ass-4-dataset/annotations/instances_val2017.json'
val_image = '/kaggle/input/cv-ass-4-dataset/val2017'

# Defining transforms for data augmentation
transform = T.Compose([T.ToTensor()])


train_dataset = CustomDataset(train_json_file, train_image, transform=transform)


validation_dataset = CustomDataset(val_json_file, val_image, transform=transform)

# Visualization of loaded datasets

image, targets = train_dataset[10]
boxes = targets["boxes"].unsqueeze(0)
labels = targets["labels"].unsqueeze(0)
image = torch.tensor(image * 255, dtype=torch.uint8)
label_str = str(labels.item())
image_with_boxes = draw_bounding_boxes(image, boxes, [label_str],colors=[(0, 255, 0)], width=4)
image_with_boxes = image_with_boxes.squeeze(0)

# Display the image with bounding boxes
plt.imshow(image_with_boxes.permute(1, 2, 0))
plt.show()

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Pad images to the maximum size in the batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    images = [torch.nn.functional.pad(img, (0, max_size[-1] - img.shape[-1], 0, max_size[-2] - img.shape[-2])) for img in images]
    for target in targets:
        target["boxes"] = target["boxes"].view(-1, 4)

    return images, targets

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=8,num_workers=2,shuffle=True,collate_fn=collate_fn)

val_dataloader = torch.utils.data.DataLoader(validation_dataset,batch_size=4,num_workers=2,shuffle=False,collate_fn=collate_fn)

# Loading pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Defining optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

# Defining learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss= 0.0
    # Wrap your data loader with tqdm for a progress bar
    for images, targets in tqdm(train_dataloader, desc=f"Processing epoch {epoch+1}"):
        # Move images and targets to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()

        # Backward pass
        losses.backward()
        optimizer.step()

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    print("Train Loss: ", train_loss)

    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    print("Validation Loss: ", val_loss)

# After all epochs, plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Saving the model
torch.save(model.state_dict(), "model_fasterRCNN.pth")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to display images with bounding boxes and scores
def show_images_side_by_side(image1, boxes1, scores1, title1, image2, boxes2, scores2, title2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display image1 with boxes1 and scores1
    ax[0].imshow(image1)
    for box, score in zip(boxes1, scores1):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(box[0], box[1], f"{score:.2f}", color='r')
    ax[0].set_title(title1)

    # Display image2 with boxes2 and scores2
    ax[1].imshow(image2)
    for box, score in zip(boxes2, scores2):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
        ax[1].text(box[0], box[1], f"{score:.2f}", color='r')
    ax[1].set_title(title2)

    plt.show()


# Set the model to evaluation mode
model.eval()

counter = 0

with torch.no_grad():
    for images, targets in val_dataloader:
        # Move images and targets to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        predictions = model(images)

        # Convert images back to PIL for visualization
        images = [transforms.ToPILImage()(img.cpu()) for img in images]

        # For each image in the batch
        for image, target, prediction in zip(images, targets, predictions):
            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()

            threshold = 0.75
            selected_indices = scores > threshold
            selected_boxes = boxes[selected_indices]
            selected_scores = scores[selected_indices]

            # Showing ground truth and prediction side by side
            show_images_side_by_side(image, target["boxes"].cpu().numpy(), target["labels"].cpu().numpy(), "Ground Truth",
                                     image, selected_boxes, selected_scores, "Prediction")

            counter += 1

            if counter >= 10:
                break

        if counter >= 10:
            break