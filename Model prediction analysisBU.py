
from datasets import Dataset, DatasetDict
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import cv2
import plotly.graph_objects as go
import glob 
import os 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def compute_score(y_true, y_pred, num_classes, id2label):
    """
    Compute Intersection over Union (IoU) and accuracy metrics for image segmentation.

    Parameters:
    - y_true (numpy array): Ground truth masks.
    - y_pred (numpy array): Predicted masks.
    - num_classes (int): Number of classes in the segmentation problem.
    - id2label (dict): Dictionary mapping class indices to class names.

    Returns:
    - results (dict): A dictionary containing IoU and accuracy metrics:
        - 'class_iou': Dictionary with class names as keys and IoU values as values.
        - 'mean_iou': Average IoU across all classes.
        - 'class_accuracy': Dictionary with class names as keys and accuracy values as values.
        - 'mean_accuracy': Average accuracy across all classes.
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape of y_true {y_true.shape}and y_pred {y_pred.shape}must match. ")

    # Initialize dictionaries to store metrics
    iou_per_class = {}
    accuracy_per_class = {}

    # Loop through each class to calculate metrics
    for cls in range(num_classes):
        
        # iou calculation
        
        class_name = id2label.get(cls, f"Class {cls}")  # Get class name using id2label
        # Calculate Intersection: Both true and predicted are the current class
        intersection = np.logical_and(y_true == cls, y_pred == cls).sum()        
        # Calculate Union: Either true or predicted is the current class
        union = np.logical_or(y_true == cls, y_pred == cls).sum()       
        # Compute IoU for this class
        iou = np.divide(intersection, union, where=union > 0)      
        iou_per_class[class_name] = np.round(iou, 3) if union > 0 else np.nan
        mean_iou = np.nanmean(list(iou_per_class.values()))

        # acc calculation
        
        # Calculate total instances of the class in y_true
        total_instances = (y_true == cls).sum()
        # Handle cases where the class is not present in y_true
        if total_instances == 0:
            accuracy_per_class[class_name] = np.nan
        else:
            # Calculate accuracy for this class
            accuracy = intersection / total_instances
            accuracy_per_class[class_name] = np.round(accuracy, 3)
        mean_accuracy = np.nanmean(list(accuracy_per_class.values()))


    # Create results dictionary
    results = {
        "class_iou": iou_per_class,
        "mean_iou": round(mean_iou, 3),
        "class_accuracy": accuracy_per_class,
        "mean_accuracy": round(mean_accuracy, 3)
    }
    
    return results

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, feature_extractor, image_size=(512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.feature_extractor = feature_extractor
        self.image_size = image_size
        self.image_transforms = Compose([Resize(image_size), ToTensor()])  # For images
        self.mask_transforms = Compose([Resize(image_size, interpolation=Image.NEAREST)])  # For masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Apply transformations
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)

        # Convert the mask to a PyTorch tensor with integer type
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Convert the image to the format required by the feature extractor
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        return inputs["pixel_values"].squeeze(0), mask

def predict_metric(selected_img, selected_mask) :
    id2label = {
        0:  'void',
        1:  'road',
        2:  'construction',
        3:  'object',
        4:  'vegetation',
        5: 'sky',
        6: 'human',
        7: 'vehicle',
    }
    label2id = { label: id for id, label in id2label.items() }

    checkpoint = "nvidia/mit-b3"  # Smallest SegFormer model
    num_classes = 8  # Define the number of classes
    feature_extractor = SegformerFeatureExtractor.from_pretrained(checkpoint, size=512)
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    lr=5e-6
    model.load_state_dict(torch.load('./model/torchB3'+str(lr)+'.pth',map_location=torch.device('cpu')))  # Load the saved parameters

    model = model.to(torch.device('cpu'))


    ds_val = SegmentationDataset(["./"+selected_img],["./"+selected_mask], feature_extractor)

    model.eval()

    # Initialize an empty list to store the predicted masks
    predicted_masks = []

    # Loop over the validation dataset (ds_val)
    with torch.no_grad():
        for _, data in enumerate(ds_val):
            # Make sure to move data to the same device as the model (if applicable)
            pixel_values = data[0].unsqueeze(0)  # Assuming pixel_values is the input
            logits = model(pixel_values=pixel_values).logits
            
            # Apply softmax to the logits if it's a multi-class segmentation problem
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
            
            # Get the predicted mask (argmax to get the class with highest probability)
            pred_mask = torch.argmax(logits, dim=1)  # Shape: [batch_size, height, width]
            
            # Convert the predicted mask to a numpy array and append to the list
            predicted_masks.append(pred_mask.squeeze(0))

    predicted_masks = np.array(predicted_masks)

    print("Predicted masks array shape:", predicted_masks.shape)

    true_masks = np.array([i[1] for i in ds_val])

    model_eval = compute_score(true_masks, predicted_masks, num_classes, id2label)

    return model_eval["class_iou"], true_masks, predicted_masks


# Define the color mappings for segmentation classes
color_class = {
    'void': [0, 0, 255],         # Blue
    'road': [255, 255, 0],       # Yellow
    'construction': [255, 0, 0], # Red
    'object': [0, 255, 255],     # Cyan
    'vegetation': [0, 255, 0],   # Green
    'sky': [255, 0, 255],        # Magenta
    'human': [255, 100, 100],    # Pale pink
    'vehicle': [255, 125, 0]     # Orange
}

# Numeric mappings for segmentation classes
num_class = {cls: idx for idx, cls in enumerate(color_class.keys())}


# Configure Streamlit app
st.set_page_config(
    page_title="Image Segmentation Dashboard",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="auto"
)


# Title and Description
st.title("Image Segmentation Dashboard")
st.write(
    """
    This dashboard provides tools for exploring image segmentation models, 
    visualizing masks, and comparing model performance across categories.
    """
)

example_images = glob.glob(os.path.join(".", '*left*'))


# Define layout
main_layout = st.columns((1, 0.2, 1), gap='small')

with main_layout[0]:
    st.header("Model inputs ")
    selected_img = st.selectbox("Select a file", example_images)
    selected_mask = "_".join(selected_img.split('_')[:3])+'_gtFine_labelIds_cat.png'
    iou_per_class_unet, true_mask, pred_mask_unet = predict_metric(selected_img,  selected_mask)
    iou_per_class_seg, true_mask, pred_mask_seg = predict_metric(selected_img,  selected_mask)
    eda_layout = st.columns((2, 2, 1), gap='medium')

    with eda_layout[0]:
        # Apply transformations and display image
        img = cv2.imread(selected_img)
        st.image(img, caption="Original Image", use_container_width=True)
    with eda_layout[1]:
        # Apply transformations and display image
        # Load images and masks
        original_image = cv2.imread(selected_img)
        mask = cv2.imread(selected_mask, cv2.IMREAD_GRAYSCALE)
        # Convert the grayscale mask to RGB
        mask_rgb = np.zeros((*original_image.shape,), dtype=np.uint8)
        for i, (cls, color) in enumerate(color_class.items()):
            mask_rgb[mask == i] = color

        combined_mask = np.zeros_like(mask_rgb)
        for cls in color_class:
            combined_mask[np.where(mask == num_class[cls])] = color_class[cls]

        # Blend original image with mask
        blended_image = cv2.addWeighted(original_image, 0.7, combined_mask, 0.3, 0)
        st.image(blended_image, caption="True mask", use_container_width=True)

### Predicted masks section
    st.header("Predicted masks")
    mask_viewer_layout = st.columns((2, 2, 1), gap='medium')

    with mask_viewer_layout[0]:
        # Load images and masks
        mask_rgb = np.zeros((*original_image.shape,), dtype=np.uint8)
        pred_mask = pred_mask_unet.squeeze(0)
        pred_mask = cv2.resize(pred_mask, (2048, 1024), interpolation=cv2.INTER_NEAREST)  # Nearest interpolation
        
        for i, (cls, color) in enumerate(color_class.items()):
            mask_rgb[pred_mask == i] = color
        st.image(mask_rgb, caption="Predicted mask - Unet", use_container_width=True)### Model Performance Section

    with mask_viewer_layout[1]:
        # Load images and masks
        mask_rgb = np.zeros((*original_image.shape,), dtype=np.uint8)
        pred_mask = pred_mask_seg.squeeze(0)
        pred_mask = cv2.resize(pred_mask, (2048, 1024), interpolation=cv2.INTER_NEAREST)  # Nearest interpolation
        
        for i, (cls, color) in enumerate(color_class.items()):
            mask_rgb[pred_mask == i] = color
        st.image(mask_rgb, caption="Predicted mask - Segformer", use_container_width=True)### Model Performance Section
    # Define colors for each class
    with mask_viewer_layout[2]:
   # Create a figure for displaying the mask
        fig, ax = plt.subplots()

        # Create a legend for class colors
        legend_patches = [Patch(color=np.array(color) / 255, label=class_name) for class_name, color in color_class.items()]
        ax.legend(handles=legend_patches, title="Classes",loc='center', fontsize=40, title_fontsize=40)
        ax.axis('off')
        # Display the plot in Streamlit
        st.pyplot(fig)



        
with main_layout[2]:
    st.header("Model Performance Evaluation")

    # Example IoU scores for models
    categories = list(color_class.keys())
    

    iou_model_1 = list(iou_per_class_unet.values())
    iou_model_2 = list(iou_per_class_seg.values())
    mean_model_1 = np.mean(iou_model_1)
    mean_model_2 = np.mean(iou_model_2)

    # IoU scores by category
    st.subheader("IoU Scores by Category")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=categories, y=iou_model_1, name="Model Unet", marker_color='blue'))
    fig1.add_trace(go.Bar(x=categories, y=iou_model_2, name="Model Segformer", marker_color='orange'))
    fig1.update_layout(
        xaxis_title="Categories",
        yaxis_title="IoU Score",
        barmode='group',
        legend_title="Models",
        template="plotly_white",
        height=300,  # Adjust height for better fit       
        margin=dict(l=20, r=20, t=0, b=20)
    )
    st.plotly_chart(fig1)

    # Mean IoU scores comparison
    st.subheader("Mean IoU Score Comparison")
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=[mean_model_1],
        y=["Model Unet"],
        orientation='h',
        marker_color='blue',
        name="Model Unet"
    ))
    fig2.add_trace(go.Bar(
        x=[mean_model_2],
        y=["Model SegFormer"],
        orientation='h',
        marker_color='orange',
        name="Model SegFormer"
    ))

    fig2.update_layout(
        xaxis_title="Mean IoU Score",
        yaxis_title="Models",
        xaxis=dict(range=[0, 1], showgrid=True),  # Set range from 0 to 1 for better alignment
        yaxis=dict(showgrid=False),  # Hide grid lines on y-axis for cleaner look
        template="plotly_white",
        height=150,  # Adjust height for better fit
        margin=dict(l=50, r=50, t=0, b=50),  # Add margins for spacing
        showlegend=False  # Hide legend as it's unnecessary for just two bars
    )

    # Add annotations for the scores
    fig2.add_annotation(
        x=mean_model_1, y="Model Unet",
        text=f"{mean_model_1:.2f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="blue",
        borderwidth=1
    )
    fig2.add_annotation(
        x=mean_model_2, y="Model SegFormer",
        text=f"{mean_model_2:.2f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="orange",
        borderwidth=1
    )

    st.plotly_chart(fig2)




### Sidebar Information
st.sidebar.header("References")
st.sidebar.write("For more information, visit:")
st.sidebar.write("[The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)")
st.sidebar.write("[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)")
st.sidebar.write("[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)")
st.sidebar.header("Accessibility Features")
st.sidebar.write(
    "This dashboard is designed to comply with WCAG standards for visualizations, including colorblind-friendly graphs."
)
