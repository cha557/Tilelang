"""CuTeDSL Source Wrapper for TileLang.
This module provides C++ kernel launcher generation for the CuTeDSL backend.
"""

from __future__ import annotations
from typing import Any, ClassVar

from tvm import IRModule
from tvm.tir.stmt_functor import post_order_visit

from tilelang import tvm as tvm
from tilelang.jit.adapter.utils import (
    pythonic_expr,
)

# =============================================================================
# C++ LAUNCHER TEMPLATES (using named parameters for clarity)
# =============================================================================

# Kernel launch template
CPP_KERNEL_LAUNCH_TEMPLATE = """\
  // Launch kernel {kernel_idx}: {kernel_name}
  {{
    // Get the kernel for current device
    auto kernels_it = g_device_kernels.find(device_id);
    if (kernels_it == g_device_kernels.end()) {{
      std::cerr << "Kernels not initialized for device " << device_id << "\\n";
      return CUDA_ERROR_NOT_INITIALIZED;
    }}
    const std::vector<CUfunction>& kernels = kernels_it->second;

    void* args[] = {{{kernel_args}}};
    result = cuLaunchKernel(
        kernels[{kernel_idx}],
        {grid_x}, {grid_y}, {grid_z},
        {block_x}, {block_y}, {block_z},
        {smem_size},
        stream,
        args,
        nullptr
    );
    if (result != CUDA_SUCCESS) {{
      std::cerr << "Failed to launch kernel {kernel_name} on device " << device_id << ": " << result << "\\n";
      return result;
    }}
  }}
"""

# Complete C++ launcher template
CPP_LAUNCHER_TEMPLATE = """\
// TVM Headers
//#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

// Main kernel launcher
int launch_kernel({launch_func_sig}, uint64_t stream, int device_id) {{
  {get_sutensor_code}
  std::cout << "launch sunmmio kernel" << std::endl;

  return 0;
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_kernel, launch_kernel);
"""

# =============================================================================
# PYTHON HOST FUNCTION TEMPLATE
# =============================================================================

PYTHON_HOST_FUNC_TEMPLATE = """\
import os
from pathlib import Path

# TODO: import tvm.ffi as ffi?
import tvm_ffi as ffi
#import tvm.ffi as ffi
#import torch_sunmmio

_cpp_launcher = None
_cpp_launcher_lib = None

class SuTensorHandle:
    def __init__(self, tensor):
        self._tensor = tensor
        #self._ptr = su_ext.get_sutensor_ptr(tensor)
        self._ptr = tensor.data_ptr()

    def __tvm_ffi_opaque_ptr__(self):
        return self._ptr

def _load_cpp_launcher():
    \"\"\"Load C++ kernel launcher.\"\"\"
    global _cpp_launcher, _cpp_launcher_lib
    if _cpp_launcher is not None:
      return _cpp_launcher
  
    lib_path = os.path.join(os.path.dirname(__file__), "{launcher_lib_name}")
    if not os.path.exists(lib_path):
      raise FileNotFoundError(f"Launcher not found: {{lib_path}}")
  
    _cpp_launcher_lib = ffi.load_module(lib_path)
    _cpp_launcher = _cpp_launcher_lib.launch_kernel
    return _cpp_launcher


def call({call_func_params}, stream, device_id=0):
    \"\"\"Kernel dispatch function.
  
    Args:
        stream: stream handle
        device_id: device ID (should be passed from caller, defaults to 0)
    \"\"\"
  
    launcher = _load_cpp_launcher()
    result = launcher({launcher_call_args} stream, device_id)
  
    if result != 0:
      raise RuntimeError(f"Kernel launch failed with error")
"""

# =============================================================================
# WRAPPER CLASS
# =============================================================================


class TLSunmmioSourceWrapper:
    """Wrapper class for TileLang Sunmmio backend with C++ launcher.

    Generates optimized C++ launcher code that:
    - Launches kernels with minimal Python overhead
    - Supports both single and multiple kernel scenarios
    """

    _TYPE_MAP: ClassVar[dict[str, str]] = {
        "float32": "cutlass.Float32",
        "float16": "cutlass.Float16",
        "bfloat16": "cutlass.BFloat16",
        "float8_e4m3": "cutlass.Float8E4M3",
        "float8_e5m2": "cutlass.Float8E5M2",
        "float64": "cutlass.Float64",
        "int64": "cutlass.Int64",
        "int32": "cutlass.Int32",
        "uint32": "cutlass.Uint32",
        "bool": "cutlass.Boolean",
        "int8": "cutlass.Int8",
        "uint8": "cutlass.Uint8",
        "int16": "cutlass.Int16",
        "uint16": "cutlass.Uint16",
        "uchar": "cutlass.Uint8",
    }

    # C++ launcher code must not depend on cutlass Python types.
    # Use plain C/C++ types for expression rendering inside generated .cpp.
    _CXX_TYPE_MAP: ClassVar[dict[str, str]] = {
        "float32": "float",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "bool": "bool",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
    }

    launcher_lib_name: str | None = "launcher_lib.so"

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.mod = scheduled_ir_module
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod = device_mod
        self.host_mod = host_mod

        # TODO:
        self.function_names: str | None = None
        # TODO: need this?
        self.parse_source_information()
        self.srcpath: str | None = None
        self.libpath: str | None = None
        self.lib_code: str | None = self.update_lib_code(source)

    def parse_source_information(self):
        assert len(self.device_mod.functions) >= 1, "Device module should have at least one function."
        assert len(self.host_mod.functions) == 1, "Only support one function in host module."

        function_names = []
        for g_var, _ in self.device_mod.functions.items():
            function_name = g_var.name_hint
            function_names.append(function_name)

        function_names_index = {}
        for g_var, func in self.host_mod.functions.items():
            function_name = g_var.name_hint
            host_code = str(func)
            for function_name in function_names:
                index = host_code.index(f'T.call_packed("{function_name}"')
                function_names_index[function_name] = index
        # sort function_names
        function_names = sorted(function_names, key=lambda x: function_names_index[x])
        self.function_names = function_names

        print("device kernel function_names:", function_names)

    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def prim_func(self):
        if len(self.mod.get_global_vars()) == 1:
            return self.mod[self.mod.get_global_vars()[0]]
        elif "main" in self.mod:
            return self.mod["main"]
        else:
            for _, function in self.mod.functions_items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")

    @property
    def device_func(self):
        if len(self.device_mod.get_global_vars()) == 1:
            return self.device_mod[self.device_mod.get_global_vars()[0]]
        elif "main" in self.device_mod:
            return self.device_mod["main"]
        else:
            for _, function in self.device_mod.functions.items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")

    @property
    def host_func(self):
        if len(self.host_mod.get_global_vars()) == 1:
            return self.host_mod[self.host_mod.get_global_vars()[0]]
        elif "main" in self.host_mod:
            return self.host_mod["main"]
        else:
            for _, function in self.host_mod.functions.items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _pythonic_expr(self, expr: tvm.tir.PrimExpr) -> str:
        """Convert TVM expression to Python string."""
        return pythonic_expr(expr, self._TYPE_MAP, floor_div_op="//")

    def _cxx_expr(self, expr: tvm.tir.PrimExpr) -> str:
        """Convert TVM expression to C++ string for generated launcher code."""
        return pythonic_expr(expr, self._CXX_TYPE_MAP)

    @staticmethod
    def _cxx_cast(ctype: str, expr_str: str) -> str:
        return f"static_cast<{ctype}>({expr_str})"

    def _collect_function_args(self) -> tuple[list[dict], list[str]]:
        """Collect all function arguments from primary function.

        Returns:
            Tuple of (function_args, buffer_args)
        """
        function_args = []
        buffer_args = []

        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({"name": buffer.data.name, "type": "buffer"})
                buffer_args.append(buffer.data.name)
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._TYPE_MAP[param.dtype]})
            else:
                raise ValueError(f"Parameter {param} not in buffer map")

        return function_args, buffer_args

    @staticmethod
    def _extract_func_call_args(
        #declaration: str,
        function_args: list[dict],
        function_params: list,
    ) -> list[tuple[str, str]]:
        """Extract function call arguments from Python function declaration."""

        # TODO: need this?
        call_args = []
        for param_name in function_params:
            for arg in function_args:
                #if arg["name"] == param_name:
                if arg["name"] == str(param_name):
                    call_args.append((param_name, arg["type"]))
        return call_args

    # =========================================================================
    # C++ Launcher Generation
    # =========================================================================

    def _generate_cpp_launcher(
        self,
        kernel_metadata_list: list[dict],
        function_args: list[dict],
    ) -> str:
        """Generate complete C++ launcher code using templates.

        TMA descriptors are stored on HOST memory in stack-local tma_descs[] array.
        cuLaunchKernel automatically copies 128-byte CUtensorMap to kernel param space
        when kernel uses __grid_constant__ parameter.
        """
        num_kernels = len(kernel_metadata_list)
        scalar_args = [arg for arg in function_args if arg["type"] != "buffer"]

        # Generate launch function signature and get_ptr code
        func_sig_parts = []
        get_sutensor_code = ""
        for arg in function_args:
            if arg["type"] == "buffer":
                func_sig_parts.append(f"void* {arg['name']}")
                get_sutensor_code += f"  // auto* su_tensor_{arg['name']} = static_cast<SuTensor*>({arg['name']});\n"

        # Generate kernel launches
        #kernel_launches = "\n".join(self._generate_kernel_launch(km, idx) for idx, km in enumerate(kernel_metadata_list))

        return CPP_LAUNCHER_TEMPLATE.format(
            #num_kernels=num_kernels,
            launch_func_sig=", ".join(func_sig_parts),
            get_sutensor_code=get_sutensor_code,
            #kernel_launches=kernel_launches,
        )

    # =========================================================================
    # Python Wrapper Generation
    # =========================================================================

    def _generate_python_wrapper(
        self,
        function_args: list[dict],
    ) -> str:
        """Generate Python wrapper code."""
        # Build function parameters
        call_func_params = ", ".join(arg["name"] for arg in function_args)
        launcher_call_args = ""
        for arg in function_args:
            if arg["type"] == "buffer":
                launcher_call_args += f"SuTensorHandle({arg['name']}),"
            else:
                launcher_call_args += f"{arg['name']},"

        return PYTHON_HOST_FUNC_TEMPLATE.format(
            launcher_lib_name=self.launcher_lib_name,
            call_func_params=call_func_params,
            launcher_call_args=launcher_call_args,
        )

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def create_dispatch_func(self, code, function_informations):
        """Create dispatch function - always use C++ launcher."""
        return self.create_dispatch_func_cpp_launcher(code, function_informations)

    def create_dispatch_func_cpp_launcher(self, code, function_informations):
        """Create dispatch function using C++ launcher."""
        # TODO need this?
        function_args, buffer_args = self._collect_function_args()
        print("function_args:", function_args)

        # Process each kernel and collect metadata
        kernel_metadata = []

        for function_name, function_info in function_informations.items():
            #declaration = extract_python_func_declaration(code, function_name)
            call_args = self._extract_func_call_args(
                #declaration,
                function_args,
                function_info["function_params"],
            )

            kernel_metadata.append(
                {
                    "function_name": function_name,
                    "function_info": function_info,
                    "call_args": call_args,
                }
            )
        print("kernel_metadata:", kernel_metadata)

        # Generate C++ launcher
        launcher_cpp_code = self._generate_cpp_launcher(
            kernel_metadata, function_args
        )

        self.launcher_cpp_code = launcher_cpp_code

        # Generate Python wrapper
        self.python_wrapper = self._generate_python_wrapper(function_args)

    def get_launcher_cpp_code(self) -> str:
        """Get the generated C++ launcher code."""
        return getattr(self, "launcher_cpp_code", "")

    def update_lib_code(self, code: str):
        """Update the library code with the given code string."""
        self.lib_code = code

        # TODO: need this? function params
        # Organize function information for code generation
        function_informations = {}
        for function_name in self.function_names:
            assert function_name in self.device_mod, f"Function {function_name} not found in device module"
            device_func = self.device_mod[function_name]
            kernel_params_cnt = len(device_func.params)
            function_params: list[str] = None

            def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
                nonlocal function_params
                if isinstance(node, tvm.tir.Call):
                    if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tir.tvm_call_packed")):
                        return
                    args = node.args
                    if not args or args[0] != fn:
                        return
                    if len(args) < 1 + param_cnt:
                        raise AssertionError("tvm_call_packed should have at least 1 argument and match device function parameters")
                    function_params = args[1 : 1 + param_cnt]

            post_order_visit(self.host_func.body, visitor)
            assert function_params is not None, "function_params should not be None"

            function_informations[function_name] = {
                "function_name": function_name,
                "function_params": function_params,
            }
            print(f"function_name: {function_name}, function_params: {function_params}")

        # create sudeck code and python wrapper
        self.create_dispatch_func(code, function_informations)