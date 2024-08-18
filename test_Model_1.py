import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision.ops import nms

def convert_to_yolo_format(boxes, image_width, image_height):
    yolo_boxes = []
    for box in boxes:
        # Calculate center x and y
        x_center = (box[0] + box[2]) / 2.0
        y_center = (box[1] + box[3]) / 2.0
        
        # Calculate width and height
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Normalize bounding box coordinates
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        yolo_boxes.append([x_center, y_center, width, height])

    return yolo_boxes

import torch

def apply_nms(boxes, scores, iou_threshold=0.1):
    # Convert numpy arrays to tensors
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)

    # Apply non-maximum suppression
    keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    # Filter detections
    filtered_boxes = boxes_tensor[keep]
    filtered_scores = scores_tensor[keep]

    return filtered_boxes.tolist(), filtered_scores.tolist()



def test_model(model, image_folder, output_folder, device):
    # Set the model to evaluation mode
    model.eval()
    
    # Define transformation to apply to images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load all images in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Extract bounding boxes and scores
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter out low confidence detections
        threshold = 0.7
        confidence_mask = scores > threshold
        filtered_boxes = boxes[confidence_mask]
        filtered_scores = scores[confidence_mask]
        
        # Apply non-maximum suppression
        iou_threshold = 0.1
        filtered_boxes, filtered_scores = apply_nms(filtered_boxes, filtered_scores, iou_threshold)
        
        # Get image width and height
        image_width, image_height = image.size
        
        # Convert bounding boxes to YOLO format
        yolo_boxes = convert_to_yolo_format(filtered_boxes, image_width, image_height)
        
        # Write predictions to text file
        output_file = os.path.splitext(image_file)[0] + "_preds.txt"
        output_path = os.path.join(output_folder, output_file)
        with open(output_path, 'w') as f:
            for box in yolo_boxes:
                # Format: center_x center_y width height confidence_score
                f.write(' '.join(map(str, box)) + ' ' + str(filtered_scores[0]) + '\n')


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  #(background and maligant)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("model_fasterRCNN.pth"))
model.to(device)
model.eval()

# Define image folder and output folder
image_folder = "D:\\Sem 2\\CV\\Assignment_4\\sample_test\\test\\images"
output_folder = "D:\\Sem 2\\CV\\Assignment_4\\RCNN_imgs"

os.makedirs(output_folder, exist_ok=True)

test_model(model, image_folder, output_folder, device)


