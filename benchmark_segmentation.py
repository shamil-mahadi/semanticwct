import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
from PIL import Image
import numpy as np
import torch
from collections import defaultdict
from typing import List
import cv2

# --- Start of original segment.py content ---

class Segment:
    def __init__(self, map, id, confidence):
        self.map = map
        self.id = id
        self.confidence = confidence

    def __repr__(self):
        return f"Map shape: {self.map.shape}, Class ID: {self.id}, Confidence: {self.confidence}"


SAM2_CKPT = "sam2.1_hiera_large.pt"
SAM2_CFG = "sam2.1_hiera_l.yaml"
GROUNDING_MODEL = "grounding-dino-tiny"

GROUNDING_BOX_THRESHOLD = 0.50
GROUNDING_TEXT_THRESHOLD = 0.50

sam2_ckpt_path = os.path.join("sam2_checkpoints", SAM2_CKPT)
sam2_cfg_path = os.path.join("configs", "sam2.1", SAM2_CFG)
grounding_model_path = f"IDEA-Research/{GROUNDING_MODEL}"

DEVICE = "cpu" # Keep as cpu for now, as per original request

# --- Timing starts for model loading ---
t0 = time.perf_counter()

sam2_predictor = SAM2ImagePredictor(
    build_sam2(sam2_cfg_path, sam2_ckpt_path, device=DEVICE)
)
t1 = time.perf_counter()
print(f"Time to load SAM2 predictor: {t1 - t0:.4f} seconds")

# sam2_predictor.model = torch.compile(sam2_predictor.model, mode="max-autotune", fullgraph=True)  # type: ignore
# t2 = time.perf_counter()
# print(f"Time to compile SAM2 model: {t2 - t1:.4f} seconds (commented out for line profiling)")


grounding_processor = AutoProcessor.from_pretrained(grounding_model_path)
t3 = time.perf_counter()
print(f"Time to load GroundingDINO processor: {t3 - t1:.4f} seconds") # Adjusted t1 if compile is commented

grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    grounding_model_path
).to(DEVICE)
t4 = time.perf_counter()
print(f"Time to load GroundingDINO model: {t4 - t3:.4f} seconds")

# grounding_model = torch.compile(grounding_model, mode="aot_eager")
# t5 = time.perf_counter()
# print(f"Time to compile GroundingDINO model: {t5 - t4:.4f} seconds (commented out for line profiling)")

# --- End of model loading timing ---


def segment_image(image_array: np.ndarray, prompt: str) -> List[Segment]:
    print("\n--- Starting segment_image function profiling ---")
    
    t_start_func = time.perf_counter()

    t0_inner = time.perf_counter()
    sam2_predictor.set_image(image_array)
    t1_inner = time.perf_counter()
    print(f"  sam2_predictor.set_image: {t1_inner - t0_inner:.4f} seconds")

    t2_inner = time.perf_counter()
    preprocessed_input = grounding_processor(
        images=image_array, text=prompt, return_tensors="pt"
    ).to(DEVICE)
    t3_inner = time.perf_counter()
    print(f"  grounding_processor preprocess: {t3_inner - t2_inner:.4f} seconds")

    t4_inner = time.perf_counter()
    with torch.inference_mode():
        grounded_output = grounding_model(**preprocessed_input)
    t5_inner = time.perf_counter()
    print(f"  grounding_model inference: {t5_inner - t4_inner:.4f} seconds")

    t6_inner = time.perf_counter()
    grounded_output = grounding_processor.post_process_grounded_object_detection(
        grounded_output,
        preprocessed_input.input_ids,
        box_threshold=GROUNDING_BOX_THRESHOLD,
        text_threshold=GROUNDING_TEXT_THRESHOLD,
        target_sizes=[image_array.shape[:2]],
    )[0]
    t7_inner = time.perf_counter()
    print(f"  grounding_processor post-process: {t7_inner - t6_inner:.4f} seconds")

    t8_inner = time.perf_counter()
    boxes = grounded_output["boxes"].cpu().numpy()
    t9_inner = time.perf_counter()
    print(f"  boxes to numpy: {t9_inner - t8_inner:.4f} seconds")

    # Early return if no detections
    if len(boxes) == 0:
        t_early_return_start = time.perf_counter()
        h, w = image_array.shape[:2]
        background_mask = np.ones((h, w), dtype=bool)
        background_segment = Segment(map=background_mask, id="bg", confidence=1.0)
        print(f"  Early return (no detections): {time.perf_counter() - t_early_return_start:.4f} seconds")
        return [background_segment]

    t10_inner = time.perf_counter()
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
        return_logits=False,
    )
    t11_inner = time.perf_counter()
    print(f"  sam2_predictor.predict (SAM inference): {t11_inner - t10_inner:.4f} seconds")

    t12_inner = time.perf_counter()
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    t13_inner = time.perf_counter()
    print(f"  masks squeeze: {t13_inner - t12_inner:.4f} seconds")

    t14_inner = time.perf_counter()
    confidences = grounded_output["scores"].cpu().numpy().tolist()
    classes = grounded_output["labels"]
    t15_inner = time.perf_counter()
    print(f"  confidences/classes extraction: {t15_inner - t14_inner:.4f} seconds")

    t16_inner = time.perf_counter()
    class_masks = defaultdict(list)
    class_confidences = defaultdict(list)
    for i in range(len(masks)):
        class_id = classes[i]
        class_masks[class_id].append(masks[i])
        class_confidences[class_id].append(confidences[i])
    t17_inner = time.perf_counter()
    print(f"  group masks by class: {t17_inner - t16_inner:.4f} seconds")

    t18_inner = time.perf_counter()
    segments = []
    combined_object_masks = []
    for class_id, masks_list in class_masks.items():
        if len(masks_list) == 1:
            combined_mask = masks_list[0]
        else:
            combined_mask = np.logical_or.reduce(masks_list)
        combined_mask = combined_mask.astype(bool)
        combined_object_masks.append(combined_mask)
        max_confidence = max(class_confidences[class_id])
        segments.append(
            Segment(map=combined_mask, id=class_id, confidence=max_confidence)
        )
    t19_inner = time.perf_counter()
    print(f"  combine masks and create segments: {t19_inner - t18_inner:.4f} seconds")

    t20_inner = time.perf_counter()
    if len(combined_object_masks) == 1:
        background_mask = ~combined_object_masks[0]
    else:
        all_objects_mask = np.logical_or.reduce(combined_object_masks)
        background_mask = ~all_objects_mask
    t21_inner = time.perf_counter()
    print(f"  create background mask: {t21_inner - t20_inner:.4f} seconds")

    t22_inner = time.perf_counter()
    background_segment = Segment(map=background_mask, id="bg", confidence=1.0)
    segments.append(background_segment)
    t23_inner = time.perf_counter()
    print(f"  add background segment: {t23_inner - t22_inner:.4f} seconds")

    t_end_func = time.perf_counter()
    print(f"--- Total segment_image function time: {t_end_func - t_start_func:.4f} seconds ---")
    return segments

# --- End of original segment.py content ---


# --- Real data for testing ---
if __name__ == "__main__":
    image_path = "content/1.jpg"
    
    # Load the image using OpenCV
    # OpenCV reads images in BGR format by default, convert to RGB
    image_array = cv2.imread(image_path)
    if image_array is None:
        print(f"Error: Could not load image from {image_path}")
        exit()
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Prompt relevant to the image (a building)
    image_prompt = "building. sky. flag."

    print(f"Running segment_image with {image_path} and prompt '{image_prompt}'...")
    segments = segment_image(image_array, image_prompt)
    print("\nSegmentation complete. Detected segments:")
    for seg in segments:
        print(seg)
