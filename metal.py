from typing import Any, cast
import ctypes, re, decimal, os, tempfile, subprocess
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.helpers import dedup, getenv, merge_dicts, PROFILE
from tinygrad.device import Buffer, ProfileGraphEntry, ProfileGraphEvent, Compiled
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.runtime.ops_metal import wait_check, to_ns_str
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support import objc
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher
from tinygrad.renderer.cstyle import MetalRenderer, CStyleLanguage

# Objective-Cのメソッド呼び出しを表現するためのヘルパー
def objc_msg(target: UOp, selector: str, *args: UOp, dtype: DType = dtypes.void) -> UOp:
    return UOp(Ops.CUSTOM, dtype, (target, *args), arg=selector)

def render_objc_msg(ctx, x: UOp) -> str:
    selector_parts = x.arg.split(":")
    if selector_parts[-1] == "": selector_parts.pop()
    
    target = ctx[x.src[0]]
    args = [ctx[src] for src in x.src[1:]]
    
    res = f"[{target} "
    for i, part in enumerate(selector_parts):
        res += f"{part}:{args[i]} "
    res = res.strip() + "]"
    return res

objc_matcher = PatternMatcher([
    (UPat(Ops.CUSTOM, name="x"), render_objc_msg)
])

class MetalHostRenderer(MetalRenderer):
    kernel_typedef = "void"
    extra_args = [] 

    def __init__(self):
        super().__init__()
        self.string_rewrite += objc_matcher

    def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
        prefix = [
            "#import <Metal/Metal.h>",
            "#import <Foundation/Foundation.h>"
        ]
        src = CStyleLanguage.render_kernel(self, function_name, kernel, bufs, uops, prefix)
        
        # autoreleasepoolハック
        src = src.replace("{", "{\n@autoreleasepool {", 1)
        src = src.rstrip()[:-1] + "}\n}"
        return src

    def render_dtype(self, dt: DType, mutable=True) -> str:
        if isinstance(dt, PtrDType): return "void*" 
        if dt == dtypes.int: return "int"
        if dt == dtypes.uint64: return "unsigned long long"
        return super().render_dtype(dt, mutable)

class MetalGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[str, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    icb_descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    icb_descriptor.setInheritBuffers(False)
    icb_descriptor.setInheritPipelineState(False)
    icb_descriptor.setMaxKernelBufferBindCount(31)

    self.icb = self.dev.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(icb_descriptor, len(jit_cache),
                                                                                                 metal.MTLResourceCPUCacheModeDefaultCache)
    if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    
    icb_label = bytes(objc.msg("UTF8String", ctypes.c_char_p)(objc.msg("description")(self.icb).retained())).decode()
    self.needs_icb_fix = int((m := re.search(r'AGXG(\d+)XFamily', icb_label)) is None or int(m.group(1)) < 15)

    self.fixedvars = merge_dicts([ji.fixedvars for ji in jit_cache])
    self.varlist = self.vars + list(self.fixedvars.keys())
    if len(self.varlist): self.int_buf = self.dev.allocator.alloc(len(self.varlist)*dtypes.int32.itemsize)

    all_pipelines, all_resources = [], [self.int_buf.buf] if len(self.varlist) else []
    for j,ji in enumerate(jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = self.icb.indirectComputeCommandAtIndex(j).retained()
      all_pipelines.append(prg._prg.pipeline_state)
      icb_command.setComputePipelineState(prg._prg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          icb_command.setKernelBuffer_offset_atIndex(b._buf.buf, b._buf.offset, i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars): icb_command.setKernelBuffer_offset_atIndex(self.int_buf.buf, self.varlist.index(v.expr)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_size), metal.MTLSize(*local_size))
      icb_command.setBarrier()

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.varlist): self.int_buf_view = self.dev.allocator._as_buffer(self.int_buf).cast('i')
    for var in self.fixedvars: self.int_buf_view[self.varlist.index(var)] = self.fixedvars[var]
    self.range = metal.NSRange(0, len(jit_cache))

    self.compile_host_launcher(input_rawbuffers)

  def compile_host_launcher(self, input_rawbuffers: list[Buffer]):
    self.input_replace_keys = list(self.input_replace.keys())
    
    i = UOp.variable("i", 0, len(self.input_replace_keys)-1)
    
    icb_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.uint64, arg=0)
    bufs_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.uint64.ptr(), arg=1)
    offsets_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=2)
    indices_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=3)
    num_inputs_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.int, arg=4)
    resources_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.uint64.ptr(), arg=5)
    num_resources_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.int, arg=6)
    queue_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.uint64, arg=7)
    range_start_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.int, arg=8)
    range_len_arg = UOp(Ops.DEFINE_GLOBAL, dtypes.int, arg=9)

    buf_val = bufs_arg.index(i).load()
    offset_val = offsets_arg.index(i).load()
    k_idx = indices_arg.index(i * 2).load()
    a_idx = indices_arg.index(i * 2 + 1).load()

    cmd = objc_msg(icb_arg, "indirectComputeCommandAtIndex:", k_idx, dtype=dtypes.uint64)
    set_buf = objc_msg(cmd, "setKernelBuffer:offset:atIndex:", buf_val, offset_val, a_idx)

    loop = UOp.range(num_inputs_arg, i, src=(set_buf,))

    cmd_buf = objc_msg(queue_arg, "commandBuffer", dtype=dtypes.uint64)
    encoder = objc_msg(cmd_buf, "computeCommandEncoder", dtype=dtypes.uint64)
    
    exec_icb = objc_msg(encoder, "executeCommandsInBuffer:withRange:", 
                       icb_arg, 
                       UOp(Ops.CUSTOM, dtypes.void, (range_start_arg, range_len_arg), arg="NSMakeRange"),
                       )
    
    end_enc = objc_msg(encoder, "endEncoding")
    commit = objc_msg(cmd_buf, "commit")
    wait = objc_msg(cmd_buf, "waitUntilCompleted")

    graph = UOp(Ops.SINK, dtypes.void, (loop, exec_icb, end_enc, commit, wait))
    
    renderer = MetalHostRenderer()
    src = renderer.render([graph])
    src = src.replace("void test(", "void launch_metal_graph(")
    src = src.replace("NSMakeRange", "NSMakeRange") 
    
    self.lib = self._compile_and_load(src)
    
    self.c_input_indices = (ctypes.c_int * (len(self.input_replace) * 2))()
    for k, (j, i) in enumerate(self.input_replace_keys):
        self.c_input_indices[k*2] = j
        self.c_input_indices[k*2+1] = i
        
    self.c_input_buffers = (ctypes.c_void_p * len(self.input_replace))()
    self.c_input_offsets = (ctypes.c_int * len(self.input_replace))()
    self.c_static_resources = (ctypes.c_void_p * len(self.all_resources))()
    for i, r in enumerate(self.all_resources):
        self.c_static_resources[i] = ctypes.cast(r, ctypes.c_void_p)
    self.c_all_resources = (ctypes.c_void_p * (len(self.all_resources) + len(self.input_replace)))()

  def _compile_and_load(self, src: str):
    with tempfile.NamedTemporaryFile(suffix=".m", delete=False) as src_file:
        src_file.write(src.encode('utf-8'))
        src_path = src_file.name
    dylib_path = src_path.replace(".m", ".dylib")
    
    try:
        subprocess.check_call([
            "clang", "-shared", "-O2", "-framework", "Metal", "-framework", "Foundation",
            "-o", dylib_path, src_path
        ])
        lib = ctypes.CDLL(dylib_path)
        lib.launch_metal_graph.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        return lib
    finally:
        pass

  def __call__(self, input_rawbuffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float|None:
    if self.command_buffer is not None and self.command_buffer in self.dev.mtl_buffers_in_flight: wait_check(self.command_buffer)
    if self.command_buffer is not None and PROFILE: self.collect_timestamps()

    for k, (j, i) in enumerate(self.input_replace_keys):
        input_idx = self.input_replace[(j, i)]
        buf = input_rawbuffers[input_idx]._buf
        self.c_input_buffers[k] = ctypes.cast(buf.buf, ctypes.c_void_p)
        self.c_input_offsets[k] = buf.offset
    
    ctypes.memmove(self.c_all_resources, self.c_static_resources, len(self.all_resources) * 8)
    current_res_idx = len(self.all_resources)
    used_input_indices = set(self.input_replace.values())
    for idx in used_input_indices:
        buf_ptr = ctypes.cast(input_rawbuffers[idx]._buf.buf, ctypes.c_void_p)
        self.c_all_resources[current_res_idx] = buf_ptr
        current_res_idx += 1

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      computeCommand = self.icb.indirectComputeCommandAtIndex(j)
      computeCommand.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_dims), metal.MTLSize(*local_dims))
    for var in self.vars: self.int_buf_view[self.varlist.index(var)] = var_vals[var]

    self.lib.launch_metal_graph(
        self.icb,
        self.c_input_buffers,
        self.c_input_offsets,
        self.c_input_indices,
        len(self.input_replace),
        self.c_all_resources,
        current_res_idx, 
        self.dev.mtl_queue,
        0, 
        len(self.jit_cache)
    )
    return None

  def collect_timestamps(self):
    st, en = decimal.Decimal(self.command_buffer.GPUStartTime()) * 1000000, decimal.Decimal(self.command_buffer.GPUEndTime()) * 1000000
    ents = [ProfileGraphEntry(self.device, cast(CompiledRunner, ji.prg)._prg.name, i, i+1, is_copy=False) for i,ji in enumerate(self.jit_cache)]
    step = (en-st)/len(ents)
    self.dev.profile_events += [ProfileGraphEvent(ents, [], [st+step*i for i in range(len(ents)+1)])]

  def __del__(self):
    if PROFILE and self.command_buffer is not None:
      wait_check(self.command_buffer)
      self.collect_timestamps()
