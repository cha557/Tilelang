from __future__ import annotations

import os
import cloudpickle
from typing_extensions import override

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class SunmmioKernelCache(KernelCache):
    # Must exist
    kernel_lib_path = "kernel_launcher.py"
    device_kernel_path = "kernel.mlir"
    host_kernel_path = "kernel_launcher.py"

    kernel_elf_path = "kernel.elf"
    launcher_lib_path = "launcher_lib.so"
    #launcher_cpp_path = "launcher.cpp"

    @override
    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func: Callable = None, verbose: bool = False):
        """
        Persists a compiled kernel to disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            kernel (JITKernel): The compiled kernel to be saved.
            func (Callable, optional): The original function.
            verbose (bool): Enable verbose log messages.

        Note:
            Saves the following files:
            - kernel.cu: The compiled kernel source code
            - wrapped_kernel.cu: The wrapped kernel source code
            - kernel_lib.so: The compiled kernel library
            - params.pkl: The serialized kernel parameters
        """
        cache_path = self._get_cache_path(key)
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

        # Save kernel source code
        try:
            self._save_kernel_source_code_to_disk(kernel, cache_path, verbose)
        except Exception:
            self.logger.exception("Error saving kernel source code to disk")

        # Save wrapped kernel source code
        try:
            self._save_wrapper_kernel_code_to_disk(kernel, cache_path, verbose)
        except Exception:
            self.logger.exception("Error saving host kernel source code to disk")

        # Save the kernel elf file
        try:
            self._save_kernel_elf_to_disk(kernel, cache_path, verbose)

        except Exception:
            self.logger.exception("Error saving kernel ELF file to disk")

        # Save the kernel launcher lib file
        try:
            self._save_kernel_launcher_lib_to_disk(kernel, cache_path, verbose)

        except Exception:
            self.logger.exception("Error saving kernel launcher lib file to disk")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, self.params_path)
            if verbose:
                self.logger.debug(f"Saving kernel parameters to disk: {params_path}")
            KernelCache._safe_write_file(params_path, "wb", lambda file: cloudpickle.dump(kernel.params, file))
        except Exception:
            self.logger.exception("Error saving kernel parameters to disk")

    def _save_kernel_elf_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        lib_gen = getattr(kernel.adapter, "lib_generator", None)
        if lib_gen and hasattr(lib_gen, "kernel_elf_path") and lib_gen.kernel_elf_path:
            kernel_elf_path = os.path.join(cache_path, self.kernel_elf_path)
            if verbose:
                self.logger.debug(f"Saving kernel ELF file to file: {kernel_elf_path}")
            KernelCache._safe_write_file(kernel_elf_path, "wb", lambda file: file.write(KernelCache._load_binary(lib_gen.kernel_elf_path)))

    def _save_kernel_launcher_lib_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        lib_gen = getattr(kernel.adapter, "lib_generator", None)
        if lib_gen and hasattr(lib_gen, "launcher_lib_path") and lib_gen.launcher_lib_path:
            launcher_lib_path = os.path.join(cache_path, self.launcher_lib_path)
            if verbose:
                self.logger.debug(f"Saving kernel launcher lib file to file: {launcher_lib_path}")
            KernelCache._safe_write_file(launcher_lib_path, "wb", lambda file: file.write(KernelCache._load_binary(lib_gen.launcher_lib_path)))