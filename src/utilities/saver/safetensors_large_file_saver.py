import json
import os
import tempfile

import torch
from typing_extensions import Generator

from src.utilities.logging import log_print

# Safetensors DType mapping (simplified from safetensors.torch)
# You might want to import _SAFETENSORS_TYPEMAP directly from safetensors.torch
# if you have it installed and want the most robust mapping.
_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    # torch.bool is not directly supported by official safetensors, often converted to U8
    # torch.complex64/128 are not supported
}


def get_safetensors_dtype(tensor: torch.Tensor) -> str:
    """Converts a torch.dtype to its safetensors string representation."""
    dtype = tensor.dtype
    if dtype not in _TORCH_TO_SAFETENSORS_DTYPE:
        raise ValueError(
            f"Unsupported dtype {dtype} for safetensors. "
            f"Supported dtypes are: {list(_TORCH_TO_SAFETENSORS_DTYPE.keys())}"
        )
    return _TORCH_TO_SAFETENSORS_DTYPE[dtype]


def save_tensors_incrementally_to_safetensors(
    tensor_generator: Generator, output_filename: str, global_metadata: dict = None
):
    """
    Saves tensors from a generator to a .safetensors file without loading all into memory.

    Args:
        tensor_generator: A generator yielding (name: str, tensor: torch.Tensor) tuples.
                          Tensors should ideally be on CPU and contiguous.
        output_filename: Path to save the .safetensors file.
        global_metadata: Optional dictionary for __metadata__ field in safetensors.
    """
    tensors_metadata = {}
    current_offset = 0

    # Use a temporary file to store raw tensor data
    # 'delete=False' is important as we'll reopen it by name for reading later.
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_data_file:
        tmp_data_filename = tmp_data_file.name

        log_print(f"Writing tensor data to temporary file: {tmp_data_filename}")
        for i, (name, tensor) in enumerate(tensor_generator):
            if not isinstance(name, str):
                raise TypeError(f"Tensor name must be a string, got {type(name)}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Tensor data must be a torch.Tensor, got {type(tensor)} for '{name}'")

            # Ensure tensor is on CPU and contiguous for .numpy().tobytes()
            if tensor.device.type != "cpu":
                tensor_cpu = tensor.cpu()
            else:
                tensor_cpu = tensor

            if not tensor_cpu.is_contiguous():
                tensor_cpu = tensor_cpu.contiguous()

            try:
                st_dtype = get_safetensors_dtype(tensor_cpu)
            except ValueError as e:
                log_print(f"Error processing tensor '{name}': {e}")
                # Clean up temporary file before raising
                tmp_data_file.close()  # Close it first before os.remove
                os.remove(tmp_data_filename)
                raise

            tensor_numpy = tensor_cpu.numpy()
            tensor_bytes = tensor_numpy.tobytes()
            num_bytes = len(tensor_bytes)

            tmp_data_file.write(tensor_bytes)

            tensors_metadata[name] = {
                "dtype": st_dtype,
                "shape": list(tensor.shape),
                "data_offsets": [current_offset, current_offset + num_bytes],
            }
            current_offset += num_bytes

            if (i + 1) % 10 == 0:  # log_print progress periodically
                log_print(
                    f"Processed tensor {i + 1}: '{name}', size: {num_bytes / (1024 * 1024):.2f} MB. Total offset: {current_offset / (1024 * 1024):.2f} MB"
                )

            # Explicitly delete tensor and cpu copy to free memory if they are large
            del tensor
            del tensor_cpu
            if "tensor_numpy" in locals():
                del tensor_numpy
            if "tensor_bytes" in locals():
                del tensor_bytes

    log_print(
        f"Finished writing data to temporary file. Total data size: {current_offset / (1024 * 1024 * 1024):.2f} GB."
    )

    # Prepare the final JSON header
    final_header_dict = {}
    if global_metadata:
        if not isinstance(global_metadata, dict):
            # Clean up temporary file before raising
            os.remove(tmp_data_filename)
            raise TypeError(f"global_metadata must be a dictionary, got {type(global_metadata)}")

        # Ensure all values in global_metadata are strings to conform to safetensors spec
        stringified_global_metadata = {}
        for k, v in global_metadata.items():
            if not isinstance(v, str):
                log_print(
                    f"Warning: Converting __metadata__ value for key '{k}' from {type(v)} to str: '{v}' -> '{str(v)}'"
                )
                stringified_global_metadata[k] = str(v)
            else:
                stringified_global_metadata[k] = v
        final_header_dict["__metadata__"] = stringified_global_metadata

    # Sort tensor metadata by name for consistent output (optional but good practice)
    # Note: The order of data in the file is determined by the generator,
    # this sorting only affects the JSON header order.
    sorted_tensors_metadata = {k: tensors_metadata[k] for k in sorted(tensors_metadata.keys())}
    final_header_dict.update(sorted_tensors_metadata)

    # header_json_bytes = json.dumps(final_header_dict, indent=2).encode("utf-8")
    # header_len = len(header_json_bytes)
    # header_len_bytes = header_len.to_bytes(
    #     8, "little"
    # )  # 8-byte little-endian header length

    header_json_bytes = json.dumps(final_header_dict, indent=None, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json_bytes)
    header_len_bytes = header_len.to_bytes(8, "little")  # 8-byte little-endian header length

    log_print(f"Writing final safetensors file: {output_filename}")
    # Write the final safetensors file
    try:
        with open(output_filename, "wb") as f_out:
            f_out.write(header_len_bytes)
            f_out.write(header_json_bytes)

            # Append data from the temporary file
            with open(tmp_data_filename, "rb") as f_data_read:
                # Buffer size for copying, e.g., 64MB
                # Adjust based on your system's memory and disk speed
                buffer_size = 64 * 1024 * 1024
                while True:
                    chunk = f_data_read.read(buffer_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
        log_print(f"Safetensors file '{output_filename}' saved successfully.")
    finally:
        # Clean up the temporary data file
        os.remove(tmp_data_filename)
        log_print(f"Temporary data file '{tmp_data_filename}' removed.")


# --- Example Usage ---
def large_tensor_generator(num_tensors=5, tensor_size_gb=0.5):
    """
    A generator that simulates loading or creating large tensors one by one.
    Each tensor will be (tensor_size_gb / 4 / (1024*1024)) x 1024 x 1024 elements of float32.
    (1 float32 = 4 bytes)
    """
    bytes_per_tensor = int(tensor_size_gb * 1024 * 1024 * 1024)
    elements_per_tensor = bytes_per_tensor // 4  # Assuming float32 (4 bytes)

    # Let's make it roughly square-ish, e.g., (dim, dim, fixed_small_dim)
    # Or for simplicity, (N, M) where N*M = elements_per_tensor
    # Example: (elements_per_tensor // 1024, 1024)
    # For very large tensors, you might hit individual dimension limits if not careful.
    # Let's use a shape that's somewhat reasonable: (X, 1024, 1024)
    # X * 1024 * 1024 = elements_per_tensor  => X = elements_per_tensor / (1024*1024)

    dim_x = elements_per_tensor // (1024 * 1024)
    if dim_x == 0 and elements_per_tensor > 0:  # If tensor is smaller than 1024*1024 floats
        dim_x = 1
        dim_y = elements_per_tensor // 1024 if elements_per_tensor // 1024 > 0 else elements_per_tensor
        dim_z = 1024 if elements_per_tensor // 1024 > 0 else 1
        if dim_y * dim_z < elements_per_tensor and dim_y == elements_per_tensor:  # single dimension
            shape = (elements_per_tensor,)
        elif dim_y * dim_z < elements_per_tensor:  # couldn't make it work, use a simpler shape
            shape = (elements_per_tensor // 1024, 1024) if elements_per_tensor >= 1024 else (elements_per_tensor,)
        else:
            shape = (dim_y, dim_z)

    elif dim_x > 0:
        shape = (dim_x, 1024, 1024)
    else:  # elements_per_tensor is 0 or very small
        shape = (100, 100)  # fallback small tensor

    actual_elements = 1
    for s_dim in shape:
        actual_elements *= s_dim
    log_print(f"Targeting tensor_size_gb: {tensor_size_gb:.2f} GB per tensor.")
    log_print(f"Calculated shape for each tensor: {shape} (approx {actual_elements * 4 / (1024**3):.2f} GB)")

    for i in range(num_tensors):
        # In a real scenario, you would load this tensor from disk,
        # from a database, or generate it via a computation.
        # It's crucial that this part doesn't load ALL tensors into memory.
        log_print(f"Generating tensor 'layer_{i}_weight'...")
        # Simulate creating a large tensor (on CPU to save GPU memory if not needed for generation)
        # For extremely large individual tensors, even this torch.randn might be an issue.
        # If so, you'd need to generate it in chunks and write those chunks if the tensor
        # itself cannot fit into memory. But this function handles multiple tensors that
        # *collectively* don't fit.
        try:
            tensor_data = torch.randn(shape, dtype=torch.float32, device="cpu")
        except RuntimeError as e:
            log_print(f"Error creating tensor of shape {shape}: {e}")
            log_print("This might be due to insufficient RAM for a single large tensor.")
            log_print("Consider reducing tensor_size_gb or using a machine with more RAM.")
            raise

        yield f"layer_{i}_weight", tensor_data

        # Simulate another smaller tensor like a bias
        if shape[0] > 1:  # Only if the first dim is not a scalar-like tensor
            bias_shape = (shape[-1],)
            log_print(f"Generating tensor 'layer_{i}_bias' with shape {bias_shape}...")
            bias_data = torch.randn(bias_shape, dtype=torch.float32, device="cpu")
            yield f"layer_{i}_bias", bias_data
            del bias_data

        log_print(f"Yielded tensor 'layer_{i}_weight' (and bias). Freeing its memory before next iteration.")
        del tensor_data  # Important to free memory if tensors are large
        torch.cuda.empty_cache()  # If tensors were ever on GPU then moved to CPU


def inspect_safetensors_header(filepath):
    """
    Inspects the header of a .safetensors file to diagnose potential issues.
    """
    print(f"Inspecting file: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        with open(filepath, "rb") as f:
            # 1. Read the 8-byte header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                print(f"Error: File is too short. Could not read 8-byte header length.")
                print(f"Read bytes: {header_len_bytes.hex() if header_len_bytes else 'None'}")
                return

            header_len = int.from_bytes(header_len_bytes, "little")
            print(f"Reported JSON header length (from first 8 bytes): {header_len} bytes")

            if header_len <= 0:
                print(f"Error: Reported header length is {header_len}, which is invalid.")
                # Peek at the next few bytes to see if they look like JSON
                peek_bytes = f.read(100)
                print(f"Next 100 bytes (or fewer): {peek_bytes}")
                try:
                    print(f"Decoded peek bytes (best effort): {peek_bytes.decode('utf-8', errors='replace')}")
                except:
                    pass
                return

            # 2. Read the JSON header string
            header_json_bytes = f.read(header_len)
            if len(header_json_bytes) < header_len:
                print(f"Error: File is truncated or header length is incorrect.")
                print(f"Expected {header_len} bytes for JSON header, but only got {len(header_json_bytes)} bytes.")
                print(f"Read JSON bytes (partial): {header_json_bytes.hex()[:200]}...")  # Print first 100 hex chars
                try:
                    print(
                        f"Decoded partial header (best effort UTF-8):\n{header_json_bytes.decode('utf-8', errors='replace')}"
                    )
                except Exception as e_decode:
                    print(f"Could not decode partial header: {e_decode}")
                return

            print(f"Successfully read {len(header_json_bytes)} bytes for JSON header.")

            # 3. Try to decode and parse the JSON
            try:
                header_str = header_json_bytes.decode("utf-8")
                print("\n--- Decoded JSON Header String ---")
                print(header_str)
                print("--- End of JSON Header String ---\n")

                parsed_header = json.loads(header_str)
                print("Successfully parsed JSON header content.")

                # Basic validation of content
                if "__metadata__" in parsed_header:
                    print("Found '__metadata__' key.")

                tensor_keys = [k for k in parsed_header if k != "__metadata__"]
                print(f"Found {len(tensor_keys)} tensor entries in header.")
                if tensor_keys:
                    print(f"First few tensor keys: {tensor_keys[:5]}")
                    for key in tensor_keys[:2]:  # Check first few tensors
                        if isinstance(parsed_header[key], dict):
                            print(f"  Tensor '{key}':")
                            for meta_key in ["dtype", "shape", "data_offsets"]:
                                if meta_key not in parsed_header[key]:
                                    print(f"    Warning: Missing '{meta_key}' for tensor '{key}'")
                                else:
                                    print(f"    '{meta_key}': {parsed_header[key][meta_key]}")
                        else:
                            print(f"    Warning: Tensor metadata for '{key}' is not a dict: {parsed_header[key]}")

            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError while decoding JSON header: {e}")
                print("The header content is not valid UTF-8.")
            except json.JSONDecodeError as e:
                print(f"json.JSONDecodeError while parsing JSON header: {e}")
                print("The header content is not valid JSON.")
                # Print context around the error
                error_pos = e.pos
                context = 200  # characters around the error
                start = max(0, error_pos - context)
                end = min(len(header_str), error_pos + context)
                print(f"\n--- Context around JSON error (char pos {error_pos}) ---")
                print(f"...{header_str[start:error_pos]} [[[ERROR AT/NEAR HERE]]] {header_str[error_pos:end]}...")
                print("--- End of Context ---")

            # 4. Check if there's data afterwards (optional, but good sanity check)
            # The total size of tensor data should match the offsets.
            # This is a more involved check. For now, just see if there's *any* data.
            remaining_data_peek = f.read(64)  # Read a bit of what should be tensor data
            if not remaining_data_peek and tensor_keys:  # If header lists tensors but no data follows
                # This can be valid if all tensors are empty.
                # Let's find the expected end of data from the last offset.
                max_offset_end = 0
                for key in tensor_keys:
                    if "data_offsets" in parsed_header[key] and len(parsed_header[key]["data_offsets"]) == 2:
                        offset_end = parsed_header[key]["data_offsets"][1]
                        if offset_end > max_offset_end:
                            max_offset_end = offset_end

                if max_offset_end > 0:  # If header implies data should exist
                    print(
                        f"Warning: Header implies data up to offset {max_offset_end}, but no more data found after header."
                    )
                else:  # Header implies no actual tensor data (e.g., all empty tensors)
                    print(
                        "Header suggests no tensor data or all tensors are empty, and no further data found (this might be OK)."
                    )

            elif remaining_data_peek:
                print(
                    f"Found {len(remaining_data_peek)} bytes of data immediately after the header (peek: {remaining_data_peek.hex()[:32]}...)."
                )
            else:
                print("No data found after header (this is OK if there are no tensors or all are empty).")

    except Exception as e:
        print(f"An unexpected error occurred during inspection: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # file_to_inspect = '/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/MUSLI_safetensors/safetensors/tile_1_4609_patch-0.img.safetensors'
    # inspect_safetensors_header(file_to_inspect)

    # --- Configuration ---
    # Total data size will be roughly num_tensors * tensor_size_gb
    # Example: 5 tensors * 0.5 GB/tensor = 2.5 GB total
    # Adjust these values based on your available disk space and desired test size
    # For hundreds of GBs, you might do:
    # NUM_TENSORS = 200
    # TENSOR_SIZE_GB_EACH = 1 # For 200GB total
    # Or:
    # NUM_TENSORS = 50
    # TENSOR_SIZE_GB_EACH = 4 # For 200GB total (individual tensors might be too large for RAM)

    NUM_TENSORS = 3  # Number of large "weight" tensors
    TENSOR_SIZE_GB_EACH = 0.1  # Approx GB for each "weight" tensor (0.1 GB = 100MB)
    OUTPUT_FILE = "large_model_streamed.safetensors"

    log_print(f"Attempting to create a safetensors file of approx {NUM_TENSORS * TENSOR_SIZE_GB_EACH:.2f} GB.")

    my_generator = large_tensor_generator(num_tensors=NUM_TENSORS, tensor_size_gb=TENSOR_SIZE_GB_EACH)

    custom_metadata = {
        "format": "pt_streamed",
        "source": "incremental_save_example",
        "total_tensors_estimate": NUM_TENSORS * 2,  # weights + biases
    }

    try:
        save_tensors_incrementally_to_safetensors(my_generator, OUTPUT_FILE, global_metadata=custom_metadata)
        log_print("-" * 30)
        log_print(f"To verify, you can try loading with: ")
        log_print(f"from safetensors.torch import load_file")
        log_print(f"tensors = load_file('{OUTPUT_FILE}')")
        log_print(f"Or for memory-mapped loading of specific tensors:")
        log_print(f"with safe_open('{OUTPUT_FILE}', framework='pt', device='cpu') as f:")
        log_print(f"  tensor_slice = f.get_slice('layer_0_weight')")
        log_print(f"  # Do something with the slice")

    except Exception as e:
        log_print(f"An error occurred during the save process: {e}")
        import traceback

        traceback.print_exc()
