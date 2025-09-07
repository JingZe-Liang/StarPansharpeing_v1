import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import toml
from openai import OpenAI
from PIL import Image

from src.utilities.io import read_image
from src.utilities.logging import log

max_tokens = 300
default_prompt = f"""
You will act as a remote sensing analyst to describe the content of the image.
Please provide a detailed description of the image in 1 to 3 sentences, including the following aspects:
1. The general content of the image, such as the type of scene (e.g., urban, rural, forest, water body, etc.).
2. The specific objects or features present in the image, such as buildings, roads, vegetation, water bodies, etc.
3. The spatial distribution of the objects or features, such as their arrangement, placements, density, and patterns.
4. Any other relevant information that can help understand the image, such as the time of capture, weather conditions, or any notable geographical feature or events.
Please ensure that your description is clear, concise, and informative, the total words should **not exceed {max_tokens}**.
Do not use the markdown format.
Do not use the bullet points or numbered lists.
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

    assert (
        img.ndim == 3 and img.dtype == np.uint8
    ), "Image must be a 3D numpy array with dtype uint8."
    img_bytes = io.BytesIO()
    Image.fromarray(img).save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return img_base64


def create_batch_template(
    img_id: str,
    img: Image.Image | np.ndarray | str,
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"],
):
    """Create a batch request template, supporting GLM and Qwen models."""

    base64_img = img_to_base64(img=img)

    if model == "glm-4v-plus":
        return {
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
        return {
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


def glm_45_v_template(img_id: str, img: Image.Image | np.ndarray | str):
    """GLM 4.5V template (backward compatible)."""
    return create_batch_template(img_id, img, "glm-4v-plus")


def qwen25_vl_template(img_id: str, img: Image.Image | np.ndarray | str):
    """Qwen 2.5 VL template (backward compatible)."""
    return create_batch_template(img_id, img, "qwen-vl-max-latest")


def create_batch_requests(
    img_paths: list[str],
    request_batch_file: str = "tmp/batch_requests.jsonl",
    model: Literal["glm-4v-plus", "qwen-vl-max-latest"] = "glm-4v-plus",
    _debug_n: int = 0,
):
    batch_requests = []
    for i, img_path in enumerate(img_paths):
        img_id = f"{i:05d}_{Path(img_path).stem}"
        img = read_image(img_path)

        # DEBUG: Log image type information
        log(f"DEBUG: Image {i} - Path: {img_path}")
        log(f"DEBUG: Image {i} - Type: {type(img)}")
        log(f"DEBUG: Image {i} - Shape: {getattr(img, 'shape', 'N/A')}")

        assert isinstance(
            img, (np.ndarray, Image.Image, str)
        ), f"Image must be a numpy array, PIL Image, or file path string, but got {type(img)}"
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
        self.status_history = []

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
        completion_window = "24h",
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


def demonstrate_architecture():
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
        "in_progress",
        "in_progress",
        "in_progress",
        "finalizing",
        "completed",
    ]

    log("Status Changes and Corresponding Query Intervals:")
    for i, status in enumerate(test_statuses):
        interval = monitor.get_next_interval(status)
        log(f"  Query {i+1} - Status: {status:12} - Interval: {interval:3} seconds")

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


def demonstrate_smart_monitoring():
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
        log(f"Query {i+1} - Status: {status:12} - Interval: {interval:3} seconds")

    log("\n=== Monitoring Features ===")
    log("1. When status remains unchanged, query interval gradually increases")
    log("2. When status changes, reset to base interval")
    log("3. Uses exponential backoff algorithm, maximum interval is 120 seconds")
    log("4. Effectively reduces API calls and CPU consumption")


# Example usage
if __name__ == "__main__":
    # Demonstrate new architecture
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
    model = "qwen-vl-max-latest"  # Can be changed to "glm-4v-plus"

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
