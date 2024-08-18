import os
import json
from PIL import Image
import numpy as np
import torch
from torchvision.ops import nms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
import torch

class CustomDataset(Dataset):
    def __init__(self, json_file, image_folder, transform=None):
        self.transform = transform
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.image_folder = image_folder
        self.images = data["images"]
        self.annotations = data["annotations"]

        # Mapping image IDs to their annotations
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

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get annotation details
        boxes = self.convert_bbox(annotation["bbox"])
        labels = annotation["category_id"]

        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([labels], dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}

    def convert_bbox(self, bbox):
        # Convert [x, y, width, height] to [x1, y1, x2, y2]
        x, y, width, height = bbox
        return [x, y, x + width, y + height]


transform = T.Compose([T.ToTensor()])

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    for target in targets:
        target["boxes"] = target["boxes"].view(-1, 4)

    return images, targets

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # Number of classes in our dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

val_json_file = 'D:\\Sem 2\\CV\\Assignment_4\\mammo_1k\\coco_1k\\annotations\\instances_val2017.json'
val_image = 'D:\\Sem 2\\CV\\Assignment_4\\mammo_1k\\coco_1k\\val2017'
validation_dataset = CustomDataset(val_json_file, val_image)

val_dataloader = torch.utils.data.DataLoader(validation_dataset,batch_size=4,shuffle=True,collate_fn=collate_fn,
)

# image, targets = validation_dataset[0]
# boxes = targets["boxes"].unsqueeze(0)  
# labels = targets["labels"].unsqueeze(0)
# image = torch.tensor(image * 255, dtype=torch.uint8)
# label_str = str(labels.item())  # Convert label to string
# image_with_boxes = draw_bounding_boxes(image, boxes, [label_str],colors=[(0, 255, 0)], width=4)
# image_with_boxes = image_with_boxes.squeeze(0)

# plt.imshow(image_with_boxes.permute(1, 2, 0))
# plt.show()


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
model.load_state_dict(torch.load("model_fasterRCNN.pth"))


# Define a function to apply NMS
def apply_nms(boxes, scores, iou_threshold=0.5):
    # Convert numpy arrays to tensors
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)

    # Apply non-maximum suppression
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

    # Filter detections
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]

    return filtered_boxes.numpy(), filtered_scores.numpy()


# Set the model to evaluation mode
model.eval()

# Initialize a counter
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
            # Process model's predictions
            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()


            # Keep only the bounding box with the highest score
            max_score_index = np.argmax(scores)
            best_box = boxes[max_score_index]
            best_label = labels[max_score_index]
            best_scores = scores[max_score_index]

            # boxes, scores = apply_nms(best_box, best_scores)

            # Show ground truth and prediction side by side
            show_images_side_by_side(image, target["boxes"].cpu().numpy(), target["labels"].cpu().numpy(), "Ground Truth",
                                     image, [best_box], [best_scores], "Prediction")


            # Increment the counter
            counter += 1

            # If we have visualized 10 images, break the loop
            if counter >= 10:
                break

        if counter >= 10:
            break



