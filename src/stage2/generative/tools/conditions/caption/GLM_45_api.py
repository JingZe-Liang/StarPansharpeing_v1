import base64
import gc
import io
import json
import os
import time
from pathlib import Path
from typing import Iterable, Literal, Optional

import jsonlines
import numpy as np
import toml
import torch
from openai import OpenAI
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utilities.io import read_image
from src.utilities.logging import log
from src.utilities.train_utils.visualization import (
    get_rgb_image,
    visualize_hyperspectral_image,
)

max_tokens = 300
default_prompt = f"""
You will act as a remote sensing analyst to describe the content of the image.
Please provide a detailed description of the image in 1 to 3 sentences, including the following aspects:
1. The general content of the remote sensing image, such as the type of scene (e.g., urban, rural, forest, water body, etc.).
2. The specific objects or features present in the image, such as buildings, roads, vegetation, water bodies, etc.
3. The spatial distribution of the objects or features, such as their arrangement, placements, density, and patterns.
4. Any other relevant information that can help understand the image, such as the time of capture, weather conditions, or any notable geographical feature or events.
Please ensure that your description is clear, concise, and informative, the total words should **not exceed {max_tokens}**.
The image provided to you is a remote sensing image from a satellite. NOT a normal photo. the photo is taken from a top-down view.
Do not use the markdown format.
Do not use the bullet points or numbered lists.
All reponse and answers must be in English, NO other languages.
Do not include any personal opinions or subjective views. 
"""


def img_to_base64(img: np.ndarray | str | Image.Image) -> str:
    """
    Convert a numpy array or PIL Image to base64 string.
    """
    if isinstance(img, str):
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(img, Image.Image):
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    assert img.ndim == 3 and img.dtype == np.uint8, (
        "Image must be a 3D numpy array with dtype uint8."
    )
    img_bytes = io.BytesIO()
    Image.fromarray(img).save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return img_base64


def create_batch_template(
    img_id: str,
    img: Image.Image | np.ndarray | str,
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"],
    prompt: Optional[str] = None,
    rgb_channels: str = "mean",
    use_linstretch: bool = True,
) -> dict:
    """Create a batch request template, supporting GLM and Qwen models."""

    if prompt is None:
        prompt = default_prompt

    base64_img, img = process_hyperspectral_image_to_base64(
        img=img, rgb_channels=rgb_channels, use_linstretch=use_linstretch
    )

    if model == "glm-4v-plus":
        temp = {
            "custom_id": img_id,
            "method": "POST",
            "url": "/v4/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": default_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_img},
                            },
                        ],
                    },
                ],
                "max_tokens": 1000,
            },
        }
    elif model == "qwen-vl-max-latest":
        temp = {
            "custom_id": img_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": default_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 1000,
            },
        }
    else:
        raise ValueError(f"Unsupported model: {model}")

    return temp


def glm_45_v_template(img_id: str, img: Image.Image | np.ndarray | str) -> dict:
    """GLM 4.5V template (backward compatible)."""
    return create_batch_template(img_id, img, "glm-4v-plus")


def qwen25_vl_template(img_id: str, img: Image.Image | np.ndarray | str) -> dict:
    """Qwen 2.5 VL template (backward compatible)."""
    return create_batch_template(img_id, img, "qwen-vl-max-latest")


def create_batch_requests(
    img_paths: list[str],
    request_batch_file: str = "tmp/batch_requests.jsonl",
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus",
    _debug_n: int = 0,
) -> None:
    batch_requests = []
    for i, img_path in enumerate(img_paths):
        img_id = f"{i:05d}_{Path(img_path).stem}"
        img = read_image(img_path)
        img = visualize_hyperspectral_image(
            img, rgb_channels="mean", use_linstretch=True
        )

        assert isinstance(img, (np.ndarray, Image.Image, str)), (
            f"Image must be a numpy array, PIL Image, or file path string, but got {type(img)}"
        )
        request = create_batch_template(img_id=img_id, img=img, model=model)
        batch_requests.append(request)
        if _debug_n > 0 and i + 1 >= _debug_n:
            break
    with open(request_batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    log(f"Saved {len(batch_requests)} requests to {request_batch_file}")
    log(f"Using model: {model}")


class SmartBatchMonitor:
    """Smart batch job monitor."""

    def __init__(self, base_interval: int = 30, max_interval: int = 300):
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.status_history: list[str] = []

    def get_next_interval(self, current_status: str) -> int:
        """Get the next query interval based on status history."""
        self.status_history.append(current_status)

        # Keep the history within a reasonable length.
        if len(self.status_history) > 10:
            self.status_history = self.status_history[-10:]

        # If the status remains unchanged for a long time, increase the query interval.
        if len(self.status_history) >= 3:
            recent_statuses = self.status_history[-3:]
            if all(status == recent_statuses[0] for status in recent_statuses):
                # Status unchanged for 3 consecutive times, increase the interval.
                consecutive_count = 0
                for i in range(len(self.status_history) - 1, 0, -1):
                    if self.status_history[i] == self.status_history[i - 1]:
                        consecutive_count += 1
                    else:
                        break

                if consecutive_count >= 2:
                    # Exponential backoff, but not exceeding the maximum interval.
                    interval = min(
                        self.base_interval * (1.5 ** (consecutive_count - 1)),
                        self.max_interval,
                    )
                    return int(interval)

        return self.base_interval

    def reset(self):
        """Reset the status history."""
        self.status_history = []


class GLM45BatchProcessor:
    """GLM 4.5V Batch Processor."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from zai import ZhipuAiClient

            env = toml.load("env.toml")
            self.client = ZhipuAiClient(api_key=api_key or env["zhipuai"]["api_key"])
        except ImportError:
            raise ImportError("Please install the zai package: pip install zhipuai")
        except KeyError:
            raise KeyError("Please configure zhipuai.api_key in env.toml")

        self.monitor = SmartBatchMonitor()

    def upload_batch_file(self, file_path: str) -> str:
        """Upload batch request file"""
        log(f"Uploading JSONL file containing request information...")
        with open(file_path, "rb") as f:
            file_object = self.client.files.create(file=f, purpose="batch")
        log(f"File uploaded successfully. Received file ID: {file_object.id}")
        batch_id = file_object.id
        assert isinstance(batch_id, str), "Batch ID should be a string"
        return batch_id

    def create_batch_job(
        self,
        input_file_id: str,
        completion_window: str = "24h",
        metadata: Optional[dict] = None,
    ) -> str:
        """Create batch job"""
        log(f"Creating Batch job based on file ID...")
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v4/chat/completions",  # tpe: ignore
            auto_delete_input_file=True,
            metadata=metadata or {},
        )
        log(f"Batch job creation completed. Received Batch job ID: {batch.id}")
        return batch.id

    def check_job_status(self, batch_id: str) -> str:
        """Check job status"""
        log(f"Checking Batch job status...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        log(f"Batch job status: {batch.status}")

        # Display progress information
        if hasattr(batch, "request_counts") and batch.request_counts:
            total = batch.request_counts.get("total", 0)
            completed = batch.request_counts.get("completed", 0)
            failed = batch.request_counts.get("failed", 0)
            if total > 0:
                progress = (completed + failed) / total * 100
                log(
                    f"Progress: {completed + failed}/{total} ({progress:.1f}%) - Success: {completed}, Failed: {failed}"
                )

        return batch.status

    def get_output_file_id(self, batch_id: str) -> Optional[str]:
        """Get output file ID"""
        log(f"Getting output file ID for successful requests in Batch job...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        log(f"Output file ID: {batch.output_file_id}")
        return batch.output_file_id

    def get_error_file_id(self, batch_id: str) -> Optional[str]:
        """Get error file ID"""
        log(f"Getting output file ID for failed requests in Batch job...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        log(f"Error file ID: {batch.error_file_id}")
        return batch.error_file_id

    def download_results(self, output_file_id: str, output_file_path: str):
        """Download results file"""
        log(f"Printing and downloading successful request results from Batch job...")
        content = self.client.files.content(output_file_id)
        log(
            f"Printing first 1000 characters of successful request results: {content.text[:1000]}..."
        )
        content.write_to_file(output_file_path)
        log(f"Complete output results saved to local output file {output_file_path}")

    def download_errors(self, error_file_id: str, error_file_path: str):
        """Download error file"""
        log(f"Printing and downloading failed request information from Batch job...")
        content = self.client.files.content(error_file_id)
        log(
            f"Printing first 1000 characters of failed request information: {content.text[:1000]}..."
        )
        content.write_to_file(error_file_path)
        log(f"Complete error information saved to local error file {error_file_path}")

    def process_batch_job(
        self,
        input_file_path: str,
        output_file_path: str = "batch_results.jsonl",
        error_file_path: str = "batch_errors.jsonl",
        completion_window: str = "24h",
        metadata: Optional[dict] = None,
        check_interval: int = 30,
    ) -> bool:
        """Complete batch job processing workflow"""
        try:
            # Step 1: Upload file
            input_file_id = self.upload_batch_file(input_file_path)

            # Step 2: Create batch job
            batch_id = self.create_batch_job(input_file_id, completion_window, metadata)

            # Step 3: Wait for job completion (smart query interval)
            self.monitor = SmartBatchMonitor(base_interval=check_interval)
            log(
                f"Starting to monitor GLM job status, base query interval: {check_interval} seconds"
            )

            while True:
                status = self.check_job_status(batch_id)

                if status in ["completed", "failed", "expired", "cancelled"]:
                    break

                # Use smart monitor to get next query interval
                next_interval = self.monitor.get_next_interval(status)
                log(f"Next query will be conducted in {next_interval} seconds...")
                time.sleep(next_interval)

            # Step 4: Process results
            if status == "failed":
                batch = self.client.batches.retrieve(batch_id=batch_id)
                log(f"Batch job failed. Error information: {batch.errors}")
                return False
            elif status in ["expired", "cancelled"]:
                log(f"Batch job has ended, status: {status}")
                return False

            # Download successful results
            output_file_id = self.get_output_file_id(batch_id)
            if output_file_id:
                self.download_results(output_file_id, output_file_path)
            else:
                log("Warning: Output file ID not found")

            # Download error information
            error_file_id = self.get_error_file_id(batch_id)
            if error_file_id:
                self.download_errors(error_file_id, error_file_path)

            return True

        except Exception as e:
            log(f"Error occurred while processing batch job: {e}")
            return False


class QwenVLBatchProcessor:
    """Qwen VL Batch Processor."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.monitor = SmartBatchMonitor()

    def upload_batch_file(self, file_path: str) -> str:
        """Upload batch request file"""
        log(f"Uploading JSONL file containing request information...")
        file_object = self.client.files.create(file=Path(file_path), purpose="batch")
        log(f"File uploaded successfully. Received file ID: {file_object.id}")
        return file_object.id

    def create_batch_job(
        self,
        input_file_id: str,
        completion_window="24h",
        metadata: Optional[dict] = None,
    ) -> str:
        """Create batch job"""
        log(f"Creating Batch job based on file ID...")
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata or {},
        )
        log(f"Batch job creation completed. Received Batch job ID: {batch.id}")
        return batch.id

    def check_job_status(self, batch_id: str) -> str:
        """Check job status"""
        log(f"Checking Batch job status...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        log(f"Batch job status: {batch.status}")

        # Display progress information
        if hasattr(batch, "request_counts") and batch.request_counts:
            total = batch.request_counts.get("total", 0)
            completed = batch.request_counts.get("completed", 0)
            failed = batch.request_counts.get("failed", 0)
            if total > 0:
                progress = (completed + failed) / total * 100
                log(
                    f"Progress: {completed + failed}/{total} ({progress:.1f}%) - Success: {completed}, Failed: {failed}"
                )

        return batch.status

    def get_output_file_id(self, batch_id: str) -> Optional[str]:
        """Get output file ID"""
        log(f"Getting output file ID for successful requests in Batch job...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        log(f"Output file ID: {batch.output_file_id}")
        return batch.output_file_id

    def get_error_file_id(self, batch_id: str) -> Optional[str]:
        """Get error file ID"""
        log(f"Getting output file ID for failed requests in Batch job...")
        batch = self.client.batches.retrieve(batch_id=batch_id)
        log(f"Error file ID: {batch.error_file_id}")
        return batch.error_file_id

    def download_results(self, output_file_id: str, output_file_path: str):
        """Download results file"""
        log(f"Printing and downloading successful request results from Batch job...")
        content = self.client.files.content(output_file_id)
        log(
            f"Printing first 1000 characters of successful request results: {content.text[:1000]}..."
        )
        content.write_to_file(output_file_path)
        log(f"Complete output results saved to local output file {output_file_path}")

    def download_errors(self, error_file_id: str, error_file_path: str):
        """Download error file"""
        log(f"Printing and downloading failed request information from Batch job...")
        content = self.client.files.content(error_file_id)
        log(
            f"Printing first 1000 characters of failed request information: {content.text[:1000]}..."
        )
        content.write_to_file(error_file_path)
        log(f"Complete error information saved to local error file {error_file_path}")

    def process_batch_job(
        self,
        input_file_path: str,
        output_file_path: str = "batch_results.jsonl",
        error_file_path: str = "batch_errors.jsonl",
        completion_window: str = "24h",
        metadata: Optional[dict] = None,
        check_interval: int = 30,
    ) -> bool:
        """Complete batch job processing workflow"""
        try:
            # Step 1: Upload file
            input_file_id = self.upload_batch_file(input_file_path)

            # Step 2: Create batch job
            batch_id = self.create_batch_job(input_file_id, completion_window, metadata)

            # Step 3: Wait for job completion (smart query interval)
            self.monitor = SmartBatchMonitor(base_interval=check_interval)
            log(
                f"Starting to monitor job status, base query interval: {check_interval} seconds"
            )

            while True:
                status = self.check_job_status(batch_id)

                if status in ["completed", "failed", "expired", "cancelled"]:
                    break

                # Use smart monitor to get next query interval
                next_interval = self.monitor.get_next_interval(status)
                log(f"Next query will be conducted in {next_interval} seconds...")
                time.sleep(next_interval)

            # Step 4: Process results
            if status == "failed":
                batch = self.client.batches.retrieve(batch_id=batch_id)
                log(f"Batch job failed. Error information: {batch.errors}")
                return False
            elif status in ["expired", "cancelled"]:
                log(f"Batch job has ended, status: {status}")
                return False

            # Download successful results
            output_file_id = self.get_output_file_id(batch_id)
            if output_file_id:
                self.download_results(output_file_id, output_file_path)
            else:
                log("Warning: Output file ID not found")

            # Download error information
            error_file_id = self.get_error_file_id(batch_id)
            if error_file_id:
                self.download_errors(error_file_id, error_file_path)

            return True

        except Exception as e:
            log(f"Error occurred while processing batch job: {e}")
            return False


def create_batch_processor(
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"],
    api_key: Optional[str] = None,
) -> GLM45BatchProcessor | QwenVLBatchProcessor:
    """
    Factory function to create a batch processor.

    Args:
        model: The model to use.
        api_key: API key (optional).

    Returns:
        An instance of the corresponding batch processor.
    """
    if model == "glm-4v-plus":
        return GLM45BatchProcessor(api_key=api_key)
    elif model == "qwen-vl-max-latest":
        return QwenVLBatchProcessor(api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model}")


def process_batch_with_model(
    img_paths: list[str],
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus",
    request_batch_file: str = "tmp/batch_requests.jsonl",
    output_file_path: str = "batch_results.jsonl",
    error_file_path: str = "batch_errors.jsonl",
    dataset_name: str = "Unknown",
    _debug_n: int = 0,
    api_key: Optional[str] = None,
) -> bool:
    """
    Process a batch of image captioning tasks using the specified model.

    Args:
        img_paths: List of image paths.
        model: The model to use ("glm-4v-plus" or "qwen-vl-max-latest").
        request_batch_file: Path to the request file.
        output_file_path: Path to the output file.
        error_file_path: Path to the error file.
        dataset_name: Dataset name (for metadata).
        _debug_n: Limit on the number of images in debug mode.
        api_key: API key (optional).

    Returns:
        bool: Whether the task completed successfully.
    """
    # Create the batch request file.
    create_batch_requests(
        img_paths=img_paths,
        request_batch_file=request_batch_file,
        model=model,
        _debug_n=_debug_n,
    )

    # Create the corresponding processor.
    processor = create_batch_processor(model, api_key=api_key)

    # Set metadata.
    metadata = {
        "description": "Hyperspectral image captioning",
        "project": f"HyperspectralTokenizer image captioning for dataset {dataset_name}",
    }

    # Process the batch job using the processor.
    return processor.process_batch_job(
        input_file_path=request_batch_file,
        output_file_path=output_file_path,
        error_file_path=error_file_path,
        completion_window="24h",
        metadata=metadata,
        check_interval=30,
    )


def demonstrate_architecture() -> None:
    """Demonstrate the new class architecture."""
    log("=== Batch Processor Architecture Demonstration ===")

    # 1. Demonstrate smart monitor
    log("\n1. Smart Monitor Demonstration:")
    monitor = SmartBatchMonitor(base_interval=10, max_interval=120)

    test_statuses = [
        "validating",
        "validating",
        "validating",
        "in_progress",
        "finalizing",
        "completed",
    ]

    log("Status Changes and Corresponding Query Intervals:")
    for i, status in enumerate(test_statuses):
        interval = monitor.get_next_interval(status)
        log(f"  Query {i + 1} - Status: {status:12} - Interval: {interval:3} seconds")

    # 2. Demonstrate processor factory
    log("\n2. Processor Factory Demonstration:")
    try:
        # Try to create GLM processor
        glm_processor = create_batch_processor("glm-4v-plus")
        log(f"  GLM Processor created successfully: {type(glm_processor).__name__}")

        # Try to create Qwen processor
        qwen_processor = create_batch_processor("qwen-vl-max-latest")
        log(f"  Qwen Processor created successfully: {type(qwen_processor).__name__}")

    except Exception as e:
        log(f"  Processor creation failed: {e}")

    # 3. Architecture features
    log("\n3. Architecture Features:")
    log("  - Unified interface design: Both processors have the same method signatures")
    log("  - Smart monitoring: Built-in SmartBatchMonitor optimizes query intervals")
    log(
        "  - Factory pattern: Create corresponding processors through create_batch_processor()"
    )
    log("  - Error handling: Unified exception handling and error information")
    log("  - Progress display: Detailed task progress and status information")

    log("\n4. Usage Examples:")
    log("  # Create processor")
    log("  processor = create_batch_processor('glm-4v-plus')")
    log("  # Or")
    log("  processor = create_batch_processor('qwen-vl-max-latest')")
    log("  ")
    log("  # Process batch job")
    log("  success = processor.process_batch_job(")
    log("      input_file_path='requests.jsonl',")
    log("      output_file_path='results.jsonl'")
    log("  )")


def demonstrate_smart_monitoring() -> None:
    """Demonstrate smart monitoring functionality (backward compatible)."""
    log("=== Smart Batch Monitoring Demonstration ===")

    # Create smart monitor
    monitor = SmartBatchMonitor(base_interval=10, max_interval=120)

    # Simulate status changes
    test_statuses = [
        "validating",
        "validating",
        "validating",
        "in_progress",
        "in_progress",
        "in_progress",
        "in_progress",
        "finalizing",
        "completed",
    ]

    log("Status Changes and Corresponding Query Intervals:")
    for i, status in enumerate(test_statuses):
        interval = monitor.get_next_interval(status)
        log(f"Query {i + 1} - Status: {status:12} - Interval: {interval:3} seconds")

    log("\n=== Monitoring Features ===")
    log("1. When status remains unchanged, query interval gradually increases")
    log("2. When status changes, reset to base interval")
    log("3. Uses exponential backoff algorithm, maximum interval is 120 seconds")
    log("4. Effectively reduces API calls and CPU consumption")


# * --- Example usage --- #


def main_api_call() -> None:
    demonstrate_architecture()

    log("\n" + "=" * 50)

    # Demonstrate smart monitoring functionality (backward compatible)
    demonstrate_smart_monitoring()

    log("\n" + "=" * 50)
    log("=== Batch Processing Example ===")

    # Configure paths
    img_dir = "data/YuZhongDataset/OpenEarthMap/OpenEarthMap_wo_xBD/chiclayo/images"
    request_batch_file = "tmp/batch_requests.jsonl"
    output_file_path = "batch_results.jsonl"
    error_file_path = "batch_errors.jsonl"
    dataset_name = "OpenEarthMap"

    # Get image paths
    img_paths = [
        str(p)
        for p in (Path(img_dir).rglob("*"))
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    ]

    log(f"Found {len(img_paths)} images")

    # Select model and process
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = (
        "qwen-vl-max-latest"  # Can be changed to "glm-4v-plus"
    )

    # Ask user if they want to run actual batch processing
    try:
        user_input = (
            input("Do you want to run actual batch processing? (y/N): ").strip().lower()
        )
    except EOFError:
        user_input = "n"
        log("\nNon-interactive environment detected, skipping batch processing.")

    if user_input == "y":
        success = process_batch_with_model(
            img_paths=img_paths,
            model=model,
            request_batch_file=request_batch_file,
            output_file_path=output_file_path,
            error_file_path=error_file_path,
            dataset_name=dataset_name,
            _debug_n=3,  # Set to 0 to process all images
        )

        if success:
            log("Batch processing completed!")
        else:
            log("Batch processing failed!")
    else:
        log(
            "Skipping batch processing, only demonstrating architecture and monitoring features."
        )

    log("\n=== New Architecture Usage Examples ===")
    log("# Direct use of processor classes")
    log("processor = GLM45BatchProcessor()  # or QwenVLBatchProcessor()")
    log("success = processor.process_batch_job('input.jsonl', 'output.jsonl')")
    log("")
    log("# Use factory function")
    log("processor = create_batch_processor('glm-4v-plus')")
    log("success = processor.process_batch_job('input.jsonl', 'output.jsonl')")


def main_prepare_json_file(
    dir_path: Optional[str] = None,
    dataloader=None,
    jsonl_file="batch_requests.jsonl",
    output_dir: Optional[str] = None,
    rgb_channels: list[int] | str = "mean",
) -> Optional[str]:
    """
    Main function: Prepare JSONL batch file
    Supports two cases: image files in directory and samples from dataloader

    Args:
        dir_path: Image directory path
        dataloader: Dataloader object
    """
    log("=== Preparing JSON File for Batch Processing ===")
    # Make ouput dir
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log(f"Output directory: {output_dir}")
    else:
        log("Output directory not specified. Using default directory.")
        output_dir = "tmp/base64/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    jsonl_file = (Path(output_dir) / jsonl_file).as_posix()

    if dir_path is not None:
        # Case 1: Prepare JSONL file from directory
        log(f"Processing images from directory: {dir_path}")

        # Configure parameters
        model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus"
        supported_extensions = [
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
            ".img",
            ".npy",
            ".mat",
        ]
        max_images = None

        # Prepare JSONL file from directory
        count = prepare_jsonl_from_directory(
            img_dir=dir_path,
            output_file=jsonl_file,
            model=model,
            rgb_channels=rgb_channels,
            supported_extensions=supported_extensions,
            max_images=max_images,
        )

        log(f"Successfully prepared {count} image requests from directory")
        return jsonl_file

    elif dataloader is not None:
        # Case 2: Prepare JSONL file from dataloader
        log("Processing samples from dataloader")

        # Configure parameters
        dataloader_model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus"
        img_key = "img"
        id_key = "__key__"
        max_batches = None

        # Prepare JSONL file from dataloader
        count = prepare_jsonl_from_dataloader(
            dataloader=dataloader,
            output_file=jsonl_file,
            model=dataloader_model,
            img_key=img_key,
            id_key=id_key,
            max_batches=max_batches,
        )

        log(f"Successfully prepared {count} sample requests from dataloader")
        return jsonl_file

    else:
        log("Error: Either dir_path or dataloader must be provided")
        return None


def prepare_jsonl_from_directory(
    img_dir: str | Path,
    output_file: str = "batch_requests.jsonl",
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus",
    prompt: Optional[str] = None,
    rgb_channels: list[int] | str = "mean",
    use_linstretch: bool = True,
    supported_extensions: Optional[list[str]] = None,
    max_images: Optional[int] = None,
) -> int:
    """
    Prepare JSONL batch file from image files in directory

    Args:
        img_dir: Image directory path
        output_file: Output JSONL file path
        model: Model to use
        prompt: Prompt text
        rgb_channels: RGB channel selection strategy
        use_linstretch: Whether to use linear stretching
        supported_extensions: Supported image file extensions
        max_images: Maximum number of images to process

    Returns:
        Number of processed images
    """
    if supported_extensions is None:
        supported_extensions = [
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
            ".img",
            ".npy",
            ".mat",
        ]

    if prompt is None:
        prompt = default_prompt

    img_dir = Path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Directory not found: {img_dir}")

    # Collect image files
    img_files: list[Path] = []
    for ext in supported_extensions:
        img_files.extend(img_dir.glob(f"*{ext}"))
        img_files.extend(img_dir.glob(f"*{ext.upper()}"))

    img_files = sorted(img_files)

    if max_images is not None and len(img_files) > max_images:
        img_files = img_files[:max_images]
        log(f"Limiting to {max_images} images")

    log(f"Found {len(img_files)} image files")

    # Create batch requests
    batch_requests = []
    for i, img_path in tqdm(enumerate(img_files)):
        img_id = f"{i:05d}_{img_path.stem}"

        request = create_batch_template(
            img_id=img_id,
            img=str(img_path),
            model=model,
            prompt=prompt,
            rgb_channels=rgb_channels,
            use_linstretch=use_linstretch,
        )
        batch_requests.append(request)
        log(f"Processing image {i + 1}/{len(img_files)}: {img_path.name}")

    # Save JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    log(f"Saved {len(batch_requests)} requests to {output_file}")
    return len(batch_requests)


def prepare_jsonl_from_dataloader(
    dataloader: DataLoader | Iterable,
    output_file: str | Path = "batch_requests.jsonl",
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus",
    prompt: Optional[str] = None,
    rgb_channels: str = "mean",
    use_linstretch: bool = True,
    img_key: str = "img",
    id_key: str = "__key__",
    max_batches: Optional[int] = None,
    max_samples_per_batch: Optional[int] = None,
    flush_every: int = 10,
) -> int:
    """
    Prepare JSONL batch file from dataloader with memory-efficient processing to avoid OOM

    Args:
        dataloader: Dataloader object
        output_file: Output JSONL file path
        model: Model to use
        prompt: Prompt text
        rgb_channels: RGB channel selection strategy
        use_linstretch: Whether to use linear stretching
        img_key: Key for image data in sample dictionary
        id_key: Key for image ID in sample dictionary
        max_batches: Maximum number of batches to process
        max_samples_per_batch: Maximum samples to process per batch (None = all)
        flush_every: Flush file to disk every N samples

    Returns:
        Number of processed samples
    """

    if prompt is None:
        prompt = default_prompt

    log(f"Processing dataloader with memory-efficient writing to avoid OOM")
    log(f"Output file: {output_file}")
    log(f"Max samples per batch: {max_samples_per_batch}")
    log(f"Flush every: {flush_every} samples")

    processed_count = 0
    batch_count = 0
    samples_since_last_flush = 0

    # Open file once and write incrementally
    id_mapping_file = (
        Path(output_file)
        .with_name("id_mappings.jsonl")
        .open(mode="w", encoding="utf-8")
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader):
            if max_batches is not None and batch_count >= max_batches:
                log(f"Reached maximum batches limit: {max_batches}")
                break

            # This is a single sample
            real_id = batch["__key__"][0]
            upload_id = str(processed_count).zfill(10)
            batch["__key__"] = [
                upload_id
            ]  # avoid the custom id not exceed 64 limitation

            if _process_and_write_sample(
                batch,
                f,
                processed_count,
                img_key,
                id_key,
                model,
                prompt,
                rgb_channels,
                use_linstretch,
            ):
                processed_count += 1
                samples_since_last_flush += 1

                id_mapping_file.write(
                    json.dumps({upload_id: real_id}, ensure_ascii=False) + "\n"
                )

                # Flush periodically
                if samples_since_last_flush >= flush_every:
                    f.flush()
                    id_mapping_file.flush()
                    samples_since_last_flush = 0

            batch_count += 1
            if batch_count % 10 == 0:
                log(f"Processed {batch_count} batches, {processed_count} samples")
                # Force cleanup after every 10 batches
                gc.collect()

    # Final flush and cleanup
    f.flush()
    gc.collect()

    log(f"Completed processing {processed_count} samples from {batch_count} batches")
    return processed_count


def _process_and_write_sample(
    sample: dict,
    file_handle,
    sample_index: int,
    img_key: str,
    id_key: str,
    model: str,
    prompt: str,
    rgb_channels: str,
    use_linstretch: bool,
) -> bool:
    """Process a single sample and write it to JSONL file with memory cleanup

    Returns True if successful, False if sample should be skipped
    """

    # Get image data
    if img_key not in sample:
        log(
            f"Image key '{img_key}' not found in sample {sample_index}, skipping",
            level="warning",
        )
        return False

    img = sample[img_key]

    # Get image ID
    if id_key in sample:
        assert len(sample[id_key]) == 1, "ID key should have a single value"
        img_id = str(sample[id_key][0])
    else:
        img_id = f"sample_{sample_index:05d}"

    # Process tensor data with immediate memory cleanup
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.squeeze(0).transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Create batch request
    request = create_batch_template(
        img_id=img_id,
        img=img,
        model=model,
        prompt=prompt,
        rgb_channels=rgb_channels,
        use_linstretch=use_linstretch,
    )

    # temp code
    # debug: save the image into a tmp dir
    # img_ = visualize_hyperspectral_image(
    #     img, to_pil=True, norm=True, rgb_channels=[3, 2, 1], use_linstretch=True
    # )
    # Path("tmp/imgs").mkdir(parents=True, exist_ok=True)
    # img_.save(f"tmp/imgs/{img_id}.jpg")

    # Write immediately to file
    file_handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    # Clear large objects from memory
    del img, request
    if sample_index % 20 == 0:  # Periodic cleanup
        gc.collect()

    return True


def prepare_jsonl_from_dataloader_samples(
    samples: list[dict],
    output_file: str = "batch_requests.jsonl",
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus",
    prompt: Optional[str] = None,
    rgb_channels: str = "mean",
    use_linstretch: bool = True,
    img_key: str = "img",
    id_key: str = "__key__",
    max_samples: Optional[int] = None,
    flush_every: int = 10,
) -> int:
    """
    Prepare JSONL batch file from sample list with memory-efficient processing

    Args:
        samples: Sample list
        output_file: Output JSONL file path
        model: Model to use
        prompt: Prompt text
        rgb_channels: RGB channel selection strategy
        use_linstretch: Whether to use linear stretching
        img_key: Key for image data in sample dictionary
        id_key: Key for image ID in sample dictionary
        max_samples: Maximum number of samples to process
        flush_every: Flush file to disk every N samples

    Returns:
        Number of processed samples
    """

    if prompt is None:
        prompt = default_prompt

    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]
        log(f"Limiting to {max_samples} samples")

    log(f"Processing {len(samples)} samples with memory-efficient writing")

    processed_count = 0
    samples_since_last_flush = 0

    # Open file once and write incrementally
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            # Get image data
            if img_key not in sample:
                log(
                    f"Image key '{img_key}' not found in sample {i}, skipping",
                    level="warning",
                )
                continue

            img = sample[img_key]

            # Get image ID
            if id_key in sample:
                img_id = str(sample[id_key])
            else:
                img_id = f"sample_{i:05d}"

                request = create_batch_template(
                    img_id=img_id,
                    img=img,
                    model=model,
                    prompt=prompt,
                    rgb_channels=rgb_channels,
                    use_linstretch=use_linstretch,
                )

                f.write(json.dumps(request, ensure_ascii=False) + "\n")
                processed_count += 1
                samples_since_last_flush += 1

                # Flush periodically to ensure data is written to disk
                if samples_since_last_flush >= flush_every:
                    f.flush()
                    samples_since_last_flush = 0

                # Clear memory
                del img, request

                # Periodic garbage collection
                if processed_count % 50 == 0:
                    gc.collect()
                    log(f"Processed {processed_count}/{len(samples)} samples")

    # Final flush and cleanup
    f.flush()
    gc.collect()

    log(f"Saved {processed_count} requests to {output_file}")
    return processed_count


def process_hyperspectral_image_to_base64(
    img: np.ndarray | str | Image.Image | Path,
    rgb_channels: str = "mean",
    use_linstretch: bool = True,
) -> str:
    """
    Convert hyperspectral image to base64 encoding

    Args:
        img: Input image, can be numpy array, file path, or PIL image
        rgb_channels: RGB channel selection strategy
        use_linstretch: Whether to use linear stretching for contrast enhancement

    Returns:
        Base64 encoded image string
    """

    # If it's a file path, read the image first
    if isinstance(img, (str, Path)):
        img_path = str(img)
        img = read_image(img_path)
        assert isinstance(img, np.ndarray), "Failed to read image file"
        log(f"Read image file: {img_path}, shape: {img.shape}")

    # Convert to RGB visualization image
    img = visualize_hyperspectral_image(
        img, rgb_channels=rgb_channels, use_linstretch=use_linstretch
    )
    img = np.squeeze(img, 0)
    assert img.ndim == 3, "Visualization image should have 3 dimensions (h, w, c)"

    # Ensure image is uint8 format
    if hasattr(img, "dtype") and img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Convert to PIL image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Convert to base64
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    base64_img = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    return base64_img, img


# * --- Batching response into txt file --- #


def batch_reponse_to_txt_files(
    batch_rs_file: str, id_mapping_file: str, output_dir: str = "tmp/batch_captions"
):
    """
    Example of the GLM response:
        {"response":{"status_code":200,"body":{"created":1759613407,"usage":{"completion_tokens":95,"prompt_tokens":2519,"total_tokens":2614},"model":"glm-4v-plus","id":"20251005053005301451c87e664e3f","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"The image depicts an urban scene, characterized by a dense arrangement of buildings and roads. Prominent features include a large circular structure, likely a stadium or arena, surrounded by smaller buildings and residential areas. The roads are organized in a grid pattern, indicating a planned urban layout. The image appears to be captured during daylight, with clear visibility of the structures and their spatial relationships. The overall scene suggests a well-developed urban area with a mix of commercial and residential zones."}}],"request_id":"0000000000"}},"custom_id":"0000000000","id":"batch_1974586444170264576"}

    """
    assert id_mapping_file.endswith(".jsonl"), (
        "ID mapping file should be in JSONL format"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load ID mappings
    id_mappings = {}
    with jsonlines.open(id_mapping_file, "r") as f:
        for item in f:
            id_mappings.update(item)

    # assert the batch response file is jsonl
    assert batch_rs_file.endswith(".jsonl"), "Batch response file should be in JSONL"
    with jsonlines.open(batch_rs_file, "r") as f:
        for rs in tqdm(f, desc="Processing batch responses ..."):
            try:
                custom_id = rs["custom_id"]
                msg = rs["response"]["body"]["choices"][0]["message"]["content"]
                mapped_id = id_mappings.get(custom_id, custom_id)
                if mapped_id == custom_id:
                    log(f"{custom_id} not found in id_mappings", level="warning")
                txt_file = Path(output_dir) / f"{mapped_id}.txt"
                with open(txt_file, "w") as f:
                    f.write(msg)
            except KeyError:
                log(f"Invalid response format, skipping: {rs}", level="warning")
                continue


if __name__ == "__main__":
    from src.data.hyperspectral_loader import get_hyperspectral_dataloaders

    # Create dataloader - note that get_hyperspectral_dataloaders returns (dataset, dataloader)
    tar_file = "data/BigEarthNet_S2/hyper_images/BigEarthNet_data_0000.tar"
    _, dl = get_hyperspectral_dataloaders(
        wds_paths=tar_file,
        batch_size=1,
        num_workers=1,
        to_neg_1_1=False,
        permute=False,
        resample=False,
        per_channel_norm=False,
    )

    # Process with memory-efficient settings to avoid OOM

    jsonl_batch_file = f"batch_file_{Path(tar_file).stem.split('_')[-1]}.jsonl"
    main_prepare_json_file(
        dataloader=dl,
        jsonl_file=jsonl_batch_file,
        output_dir="data/BigEarthNet_S2",
        rgb_channels=[3, 2, 1],
    )

    # batch_reponse_to_txt_files(
    #     "output_202510050533.jsonl",
    #     "id_mappings.jsonl",
    # )
