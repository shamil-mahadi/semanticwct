import torch
import os
from wavelet_model import WaveEncoder, WaveDecoder
from wct import perform_wct
from segmentation import segment_image, Segment
import numpy as np
import cv2
import io
from typing import List, Tuple, Dict
from cv2.ximgproc import guidedFilter


class TransferPipeline:
    def __init__(self, ckpt_dir: str = "ckpts/"):
        self.device = "cpu"
        self.unpool_method = "sum"
        self.encoder = WaveEncoder(self.unpool_method).to(self.device)
        self.decoder = WaveDecoder(self.unpool_method).to(self.device)
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(
                    ckpt_dir, "wave_encoder_{}_l4.pth".format(self.unpool_method)
                ),
                map_location=torch.device("cpu"),
            )
        )
        self.decoder.load_state_dict(
            torch.load(
                os.path.join(
                    ckpt_dir, "wave_decoder_{}_l4.pth".format(self.unpool_method)
                ),
                map_location=torch.device("cpu"),
            )
        )

        self.encoder = torch.compile(self.encoder, mode="max-autotune")
        self.decoder = torch.compile(self.decoder, mode="max-autotune")

        self.encode = self.encoder.encode  # type: ignore
        self.decode = self.decoder.decode  # type: ignore

    def get_all_feature(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, torch.Tensor]]:
        skips = {}
        feats = {"encoder": {}, "decoder": {}}
        for level in range(1, 5):
            x = self.encode(x, skips, level)

        feats["decoder"][4] = x
        for level in range(4, 1, -1):
            x = self.decode(x, skips, level)
            feats["decoder"][level - 1] = x

        return feats, skips

    def _transfer(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        content_masks: List[Segment] | None,
        style_masks: List[Segment] | None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        content_skips = {}
        style_feats, style_skips = self.get_all_feature(style)

        for level in range(1, 5):
            content = self.encode(content, content_skips, level)

        for level in ["pool1", "pool2", "pool3"]:
            for component in range(3):
                content_skips[level][component] = perform_wct(
                    content_skips[level][component],
                    style_skips[level][component],
                    content_masks,
                    style_masks,
                    alpha=alpha,
                )

        for level in range(4, 0, -1):
            content = perform_wct(
                content,
                style_feats["decoder"][level],
                content_masks,
                style_masks,
                alpha=alpha,
            )
            content = self.decode(content, content_skips, level)
        return content

    def _preprocess_img(self, img: np.ndarray, size: int | None = None) -> np.ndarray:
        if size:
            h, w = img.shape[:2]

            # Calculate 16-pixel aligned dimensions that preserve aspect ratio
            if h > w:
                new_h = size
                new_w = int((w / h) * size)
            else:
                new_w = size
                new_h = int((h / w) * size)

            # Round to nearest 16-pixel alignment
            new_h = (new_h // 16) * 16
            new_w = (new_w // 16) * 16

            # Ensure minimum size
            if new_h < 16:
                new_h = 16
            if new_w < 16:
                new_w = 16

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return img

    def _cv2_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        return img_tensor

    def transfer(
        self,
        content_bytes: bytes,
        style_bytes: bytes,
        grounding_prompt: str | None = None,
        strength: float = 1.0,
        do_segmentation: bool = False
    ) -> bytes:
        content_img = np.frombuffer(content_bytes, np.uint8)
        style_img = np.frombuffer(style_bytes, np.uint8)

        content_img = cv2.imdecode(content_img, cv2.IMREAD_COLOR_RGB)
        style_img = cv2.imdecode(style_img, cv2.IMREAD_COLOR_RGB)

        original_size = content_img.shape[:2][::-1]
        original_content_img = content_img

        content_img = self._preprocess_img(content_img, size=512)
        style_img = self._preprocess_img(style_img, size=512)

        print("DONE PREPROCESS")

        if do_segmentation:
            assert grounding_prompt != None
            
            content_masks = segment_image(content_img, grounding_prompt)
            style_masks = segment_image(style_img, grounding_prompt, filter_segments=content_masks)
            print("CHECK", content_masks, style_masks)
            print("DONE SEGMENT")
        else:
            content_masks = style_masks = None


        content_tensor = self._cv2_to_tensor(content_img)
        style_tensor = self._cv2_to_tensor(style_img)

        with torch.inference_mode():
            result = self._transfer(
                content_tensor, style_tensor, content_masks, style_masks, strength
            )

        print("DONE TRANSFER")

        result = result.clamp(0, 1).squeeze(0)
        result = result.permute(1, 2, 0).cpu().numpy()
        result_array = (result * 255).astype(np.uint8)

        result_array = cv2.resize(
            result_array, original_size, interpolation=cv2.INTER_LANCZOS4
        )

        r = (original_size[0] // 512) * 100
        eps = (0.02 * 255) ** 2
        smooth_result = guidedFilter(
            guide=original_content_img, src=result_array, radius=r, eps=eps
        )
        smooth_result = cv2.cvtColor(smooth_result, cv2.COLOR_RGB2BGR)
        _, smooth_result = cv2.imencode(".png", smooth_result)
        smooth_result_bytes = smooth_result.tobytes()
        return smooth_result_bytes


import os
from tqdm import tqdm

if __name__ == "__main__":
    pipeline = TransferPipeline()
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Process images 1 through 10
    for i in tqdm(range(1, 2), desc="Processing images"):
        content_path = f"content/{i}.jpg"
        style_path = f"styles/{i}.jpg"
        output_path = f"outputs/{i}.png"
        
        # Check if both content and style files exist
        if os.path.exists(content_path) and os.path.exists(style_path):
            print(f"Processing image {i}...")
            
            try:
                # Perform style transfer with proper resource management
                with open(content_path, "rb") as content_file, open(style_path, "rb") as style_file:
                    out = pipeline.transfer(
                        content_file.read(),
                        style_file.read(),
                         do_segmentation=True,
                         grounding_prompt="sky. mountain.",
                        strength=1.0
                    )
                
                # Write output
                with open(output_path, "wb") as f:
                    f.write(out)
                
                print(f"✓ Successfully processed {i}.jpg -> {output_path}")
                
            except Exception as e:
                print(f"✗ Error processing image {i}: {e}")
        else:
            print(f"⚠ Skipping image {i} - missing content or style file")
    
    print("Batch processing complete!")
