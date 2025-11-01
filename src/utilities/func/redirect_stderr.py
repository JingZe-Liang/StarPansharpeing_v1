"""
Utility for redirecting stdout and stderr to files with filtering.

This module provides a context manager for redirecting stdout and stderr to files
while filtering out specific warnings like libpng warnings and loguru debug messages.
"""

import os
import sys
from contextlib import contextmanager
from typing import Any, Generator


@contextmanager
def redirect_stdout_stderr_to_file(
    output_file: str = "tmp/stderr.txt", filter_patterns: list[str] | None = None
) -> Any:
    """
    Context manager for redirecting both stdout and stderr to a file with filtering.

    This context manager redirects both stdout and stderr to a specified file, allowing
    you to capture and filter out unwanted warnings during execution. This is useful for
    filtering out libpng warnings that are printed directly to stdout.

    Parameters
    ----------
    output_file : str, optional
        Path to the output file where stdout and stderr will be written.
        Default is 'tmp/stderr.txt'.
    filter_patterns : list[str] | None, optional
        List of string patterns to filter out from output.
        If None, no filtering is applied.

    Yields
    ------
    Any
        Context manager for use in 'with' statements.

    Example
    -------
    >>> with redirect_stdout_stderr_to_file('error_log.txt', ['libpng warning']):
    ...     # Your code that generates output
    ...     pass
    """
    if filter_patterns is None:
        filter_patterns = []

    # Store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Open the output file
        with open(output_file, "w") as output_file_obj:
            # Create a custom file-like object for filtering
            class FilteredWriter:
                def __init__(self, file_obj: Any) -> None:
                    self.file_obj = file_obj

                def write(self, text: str) -> int:
                    # Apply filtering if patterns are specified
                    if filter_patterns:
                        for pattern in filter_patterns:
                            if pattern in text:
                                return 0  # Skip writing this line
                    return self.file_obj.write(text)

                def flush(self) -> None:
                    self.file_obj.flush()

                def close(self) -> None:
                    self.file_obj.close()

            # Create filtered writer
            filtered_writer = FilteredWriter(output_file_obj)

            # Redirect both stdout and stderr
            sys.stdout = filtered_writer
            sys.stderr = filtered_writer

            # Yield control back to the caller
            yield filtered_writer

    finally:
        # Always restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@contextmanager
def redirect_stderr_to_file(
    output_file: str = "tmp/stderr.txt", filter_patterns: list[str] | None = None
) -> Any:
    """
    Context manager for redirecting stderr to a file with optional filtering.

    This context manager redirects stderr to a specified file, allowing
    you to capture and filter out unwanted warnings during execution.

    Parameters
    ----------
    output_file : str, optional
        Path to the output file where stderr will be written.
        Default is 'tmp/stderr.txt'.
    filter_patterns : list[str] | None, optional
        List of string patterns to filter out from stderr output.
        If None, no filtering is applied.

    Yields
    ------
    Any
        Context manager for use in 'with' statements.

    Example
    -------
    >>> with redirect_stderr_to_file('error_log.txt', ['libpng warning']):
    ...     # Your code that generates stderr output
    ...     pass
    """
    if filter_patterns is None:
        filter_patterns = []

    # Store original stderr
    original_stderr = sys.stderr

    try:
        # Open the output file
        with open(output_file, "w") as stderr_file:
            # Create a custom file-like object for filtering
            class FilteredWriter:
                def __init__(self, file_obj: Any) -> None:
                    self.file_obj = file_obj

                def write(self, text: str) -> int:
                    # Apply filtering if patterns are specified
                    if filter_patterns:
                        for pattern in filter_patterns:
                            if pattern in text:
                                return 0  # Skip writing this line
                    return self.file_obj.write(text)

                def flush(self) -> None:
                    self.file_obj.flush()

                def close(self) -> None:
                    self.file_obj.close()

            # Create filtered writer
            filtered_writer = FilteredWriter(stderr_file)

            # Redirect stderr
            sys.stderr = filtered_writer

            # Yield control back to the caller
            yield filtered_writer

    finally:
        # Always restore original stderr
        sys.stderr = original_stderr


def create_standard_redirect_context(output_file: str = "tmp/stderr.txt") -> Any:
    """
    Create a standard stdout+stderr redirect context with common filters.

    This is a convenience function that creates a redirect context for both stdout and stderr
    with commonly used filters such as libpng warnings and debug messages.

    Parameters
    ----------
    output_file : str, optional
        Path to the output file where stdout and stderr will be written.
        Default is 'tmp/stderr.txt'.

    Returns
    -------
    Any
        Context manager for use in 'with' statements.

    Example
    -------
    >>> with create_standard_redirect_context():
    ...     # Your code that generates output here
    ...     pass
    """
    # Common filter patterns including loguru debug messages
    filter_patterns = ["libpng warning: iCCP: known incorrect sRGB profile"]

    return redirect_stdout_stderr_to_file(output_file, filter_patterns)


@contextmanager
def suppress_stdout_stderr() -> Generator[None, None, None]:
    """Context manager to suppress stdout and stderr by redirecting to /dev/null."""
    # Save original file descriptors
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)

    try:
        # Open /dev/null
        devnull = os.open("/dev/null", os.O_WRONLY)
        # Redirect stdout and stderr to /dev/null
        os.dup2(devnull, 1)  # stdout
        os.dup2(devnull, 2)  # stderr
        os.close(devnull)

        yield  # Execute code in the context

    finally:
        # Restore original stdout and stderr
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
