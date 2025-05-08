from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import cv2
import numpy as np
import torch
from collections import defaultdict
from typing import List, Optional


class Segment:
    def __init__(self, map, id, confidence):
        self.map = map
        self.id = id
        self.confidence = confidence

    def __repr__(self):
        return f"Map shape: {self.map.shape}, Class ID: {self.id}, Confidence: {self.confidence}"


SAM2_CKPT = "sam2.1_hiera_large.pt"
SAM2_CFG = "sam2.1_hiera_l.yaml"
GROUNDING_MODEL = "grounding-dino-base"

GROUNDING_BOX_THRESHOLD = 0.3
GROUNDING_TEXT_THRESHOLD = 0.3

sam2_ckpt_path = os.path.join("sam2_checkpoints", SAM2_CKPT)
sam2_cfg_path = os.path.join("configs", "sam2.1", SAM2_CFG)
grounding_model_path = f"IDEA-Research/{GROUNDING_MODEL}"

DEVICE = "cpu"

sam2_predictor = SAM2ImagePredictor(
    build_sam2(sam2_cfg_path, sam2_ckpt_path, device=DEVICE)
)

sam2_predictor.model = torch.compile(sam2_predictor.model, mode="max-autotune", fullgraph=True)  # type: ignore

grounding_processor = AutoProcessor.from_pretrained(grounding_model_path)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    grounding_model_path
).to(DEVICE)

# grounding_model = torch.compile(grounding_model, mode="aot_eager")

numz = 1

def segment_image(image_array: np.ndarray, prompt: str, filter_segments: Optional[List[Segment]] = None) -> List[Segment]:
    sam2_predictor.set_image(image_array)

    preprocessed_input = grounding_processor(
        images=image_array, text=prompt, return_tensors="pt"
    ).to(DEVICE)
    with torch.inference_mode():
        grounded_output = grounding_model(**preprocessed_input)

    grounded_output = grounding_processor.post_process_grounded_object_detection(
        grounded_output,
        preprocessed_input.input_ids,
        box_threshold=GROUNDING_BOX_THRESHOLD,
        text_threshold=GROUNDING_TEXT_THRESHOLD,
        target_sizes=[image_array.shape[:2]],
    )[0]

    boxes = grounded_output["boxes"].cpu().numpy()

    # Early return if no detections
    if len(boxes) == 0:
        h, w = image_array.shape[:2]
        background_mask = np.ones((h, w), dtype=bool)
        background_segment = Segment(map=background_mask, id="bg", confidence=1.0)
        print([background_segment])
        return [background_segment]

    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
        return_logits=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = grounded_output["scores"].cpu().numpy().tolist()
    classes = grounded_output["labels"]

    # Group masks by class
    class_masks = defaultdict(list)
    class_confidences = defaultdict(list)

    for i in range(len(masks)):
        class_id = classes[i]
        class_masks[class_id].append(masks[i])
        class_confidences[class_id].append(confidences[i])

    # Create combined segments for each class
    segments = []
    combined_object_masks = []

    for class_id, masks_list in class_masks.items():
        # Optimize: avoid extra work for single masks
        if len(masks_list) == 1:
            combined_mask = masks_list[0]
        else:
            # Combine multiple masks for this class using logical OR
            combined_mask = np.logical_or.reduce(masks_list)

        # Ensure mask is boolean type
        combined_mask = combined_mask.astype(bool)

        # Use the highest confidence score for this class
        max_confidence = max(class_confidences[class_id])

        segments.append(
            Segment(map=combined_mask, id=class_id, confidence=max_confidence)
        )

    # Apply filtering if filter_segments is provided
    if filter_segments is not None:
        # Get set of allowed class IDs (excluding background)
        allowed_ids = {seg.id for seg in filter_segments if seg.id != "bg"}
        
        # Filter segments to keep only those with IDs in the filter list
        filtered_segments = []
        rejected_masks = []
        
        for segment in segments:
            if segment.id in allowed_ids:
                filtered_segments.append(segment)
            else:
                rejected_masks.append(segment.map)
        
        # Remove overlapping areas from lower confidence segments
        IoU_THRESHOLD = 0.5
        
        # Sort segments by confidence (highest first) to process in order
        filtered_segments.sort(key=lambda x: x.confidence, reverse=True)
        
        # Process overlaps - higher confidence masks take precedence
        for i in range(len(filtered_segments)):
            for j in range(i + 1, len(filtered_segments)):
                seg_high = filtered_segments[i]  # Higher confidence (processed first)
                seg_low = filtered_segments[j]   # Lower confidence
                
                # Skip if same class ID (they're already combined)
                if seg_high.id == seg_low.id:
                    continue
                
                # Calculate IoU
                intersection = np.logical_and(seg_high.map, seg_low.map)
                union = np.logical_or(seg_high.map, seg_low.map)
                
                if np.sum(intersection) == 0:
                    continue
                
                iou = np.sum(intersection) / np.sum(union)
                
                if iou > IoU_THRESHOLD:
                    # Remove intersecting area from lower confidence mask
                    seg_low.map = np.logical_and(seg_low.map, ~intersection)
        
        # Remove segments that became empty after overlap processing
        filtered_segments = [seg for seg in filtered_segments if np.any(seg.map)]
        
        # Collect object masks for background calculation
        filtered_object_masks = [seg.map for seg in filtered_segments]
        
        # Create background mask from:
        # 1. Areas not covered by any filtered objects
        # 2. Rejected masks (objects unique to this image)
        if len(filtered_object_masks) == 0:
            # No filtered objects, entire image is background
            h, w = image_array.shape[:2]
            background_mask = np.ones((h, w), dtype=bool)
        else:
            # Areas not covered by filtered objects
            all_filtered_objects = np.logical_or.reduce(filtered_object_masks)
            background_mask = ~all_filtered_objects
            
            # Add rejected masks to background
            if len(rejected_masks) > 0:
                rejected_combined = np.logical_or.reduce(rejected_masks)
                background_mask = np.logical_or(background_mask, rejected_combined)
        
        segments = filtered_segments
        combined_object_masks = filtered_object_masks
    else:
        # No filtering - use all detected objects
        for segment in segments:
            combined_object_masks.append(segment.map)
        
        # Create background mask by inverting all object masks
        if len(combined_object_masks) == 1:
            background_mask = ~combined_object_masks[0]
        else:
            all_objects_mask = np.logical_or.reduce(combined_object_masks)
            background_mask = ~all_objects_mask

    # Add background segment
    background_segment = Segment(map=background_mask, id="bg", confidence=1.0)
    segments.append(background_segment)

    global numz
    for segment in segments:
        cv2.imwrite(f"{numz}_{segment.id}.png", segment.map.astype(np.uint8) * 255)
    numz +=1
    return segments