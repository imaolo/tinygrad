from __future__ import annotations
import ctypes
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, Deque, Tuple
from tinygrad.helpers import DType, dtypes, prod, GlobalCounters, ImageDType
from weakref import ref, WeakSet, ReferenceType
class RawBuffer:  # pylint: disable=abstract-method
  live_buffers: Dict[Any, WeakSet[RawBuffer]] = defaultdict(WeakSet)
  live_buffers_ordered: Dict[Any, Deque[ReferenceType[RawBuffer]]]  = defaultdict(deque)
  def __init__(self, size:int, dtype:DType, buf:Any=None, allocator:Any=None, **kwargs):
    self.size: int = size
    self.dtype: DType = dtype
    self.offset: int = 0    # TODO: this is very unsupported, only in disk
    self._device = kwargs.get('device', None)
    self._memsz: int = size*dtype.itemsize
    self._cpu_back: np.ndarray = None
    self.kwargs = kwargs
    alloc_device = self._device if self._device is not None else '0'
    try: self.__buf = buf if buf is not None else (allocator(size, dtype, **kwargs) if allocator else None) # If buf is provided, use it. Otherwise try to allocate from the allocator.
    except: # allocation failed, swap time
      while allocator._get_cur_free_space(alloc_device) < self._memsz and len(self.live_buffers_ordered[alloc_device]):
        swap_buf_ref = self.live_buffers_ordered[alloc_device].popleft()
        if swap_buf_ref() is not None and swap_buf_ref() in self.live_buffers[alloc_device]:
          print("swappin'")
          swap_buf: RawBuffer = swap_buf_ref()
          self.live_buffers[alloc_device].remove(swap_buf)
          assert swap_buf._cpu_back is None
          swap_buf._cpu_back = swap_buf.toCPU().copy()
          allocator.free(swap_buf.__buf)
        try: allocator.ensure_has_free_space(self._memsz, alloc_device)
        except: pass
      self.__buf = allocator(size, dtype, **kwargs)
    self._allocator = allocator
    self.live_buffers[alloc_device].add(self)
    self.live_buffers_ordered[alloc_device].append(ref(self))
    GlobalCounters.mem_used += self._memsz
  @property
  def _buf(self) -> Any:
    if self._cpu_back is None: return self.__buf
    alloc_device = self._device if self._device is not None else '0'
    print("swap back")
    try: self.__buf = self._allocator(self.size, self.dtype, **self.kwargs)
    except: # MORE SWAP!
      while self._allocator._get_cur_free_space(alloc_device) < self._memsz and len(self.live_buffers_ordered[alloc_device]):
        swap_buf_ref = self.live_buffers_ordered[alloc_device].popleft()
        if swap_buf_ref() is not None and swap_buf_ref() in self.live_buffers[alloc_device]:
          print("swappin', but nested")
          swap_buf: RawBuffer = swap_buf_ref()
          self.live_buffers[alloc_device].remove(swap_buf)
          assert swap_buf._cpu_back is None
          swap_buf._cpu_back = swap_buf.toCPU().copy()
          self._allocator.free(swap_buf.__buf)
        try: self._allocator.ensure_has_free_space(self._memsz, alloc_device)
        except: pass
      self.__buf = self._allocator(self.size, self.dtype, **self.kwargs)
    recursion_police = self._cpu_back
    self._cpu_back = None
    self._copyin(recursion_police)
    self.live_buffers[alloc_device].add(self)
    self.live_buffers_ordered[alloc_device].append(ref(self))
    return self.__buf
  @_buf.setter
  def _buf(self, val): self.__buf = val
  def __del__(self):  # NOTE: if it fails on init (bad dtype), it won't have a _memsz
    if hasattr(self, '_memsz'): GlobalCounters.mem_used -= self._memsz
    if hasattr(self, '_allocator') and self._allocator and self._cpu_back is None: self._allocator.free(self.__buf)
    self._cpu_back = None
  def __repr__(self): return f"buffer<{self.size}, {self.dtype}, {id(self)}>"
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs):
    ret = cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
    if x.size > 0: ret._copyin(x)
    return ret
  def _copyin(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")

class RawBufferMapped(RawBuffer):
  def _buffer(self) -> memoryview: raise NotImplementedError("must be implemented")
  # NOTE: this metadata prevents the backing buffer from being freed. hack can be removed with PEP688
  def toCPU(self) -> np.ndarray: return np.frombuffer(self._buffer(), dtype=np.dtype(self.dtype.np, metadata={"backing": self}), count=self.size)
  def _copyin(self, x:np.ndarray) -> None: np.copyto(self.toCPU(), x.reshape(-1))

# this one is simple enough that i moved it out of the runtimes
ctypes_map = {dtypes.float64:ctypes.c_double, dtypes.float32: ctypes.c_float, dtypes.float16: ctypes.c_int16, dtypes.bfloat16: ctypes.c_int16, dtypes.int8: ctypes.c_int8, dtypes.uint8: ctypes.c_uint8, dtypes.bool: ctypes.c_uint8, dtypes.int32: ctypes.c_int32, dtypes.uint32: ctypes.c_uint32, dtypes.int64: ctypes.c_int64, dtypes.uint64: ctypes.c_uint64, dtypes.int16: ctypes.c_int16, dtypes.uint16: ctypes.c_uint16}
class RawMallocBuffer(RawBufferMapped):
  def __init__(self, size, dtype: DType): super().__init__(size, dtype, (ctypes_map[dtype] * size)())
  def _buffer(self): return memoryview(self._buf)

class RawBufferCopyInOut(RawBuffer):
  def _copyout(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  def toCPU(self) -> np.ndarray:
    x: np.ndarray = np.empty(self.size, dtype=self.dtype.np)
    if x.size > 0: self._copyout(x)
    return x

class RawBufferTransfer(RawBuffer):
  def _transfer(self, x:RawBuffer) -> None: raise NotImplementedError("must be implemented")

class LRUAllocator:
  def __init__(self, dev_memsz=(4<<30)):
    self.epoch = 0
    self.free_space: Dict[Any, int] = defaultdict(lambda: dev_memsz)
    self.buffer_info: Dict[Any, Tuple[int, DType, str]] = dict()
    self.cached_buffers: Dict[Tuple[int, ...], Deque[Tuple[Any, int]]] = defaultdict(deque) # Cached buffer storage, splitted by type and size, newest first.
    self.aging_order: Dict[Any, Deque[Tuple[Tuple[int, ...], int]]] = defaultdict(deque) # Keys of cached_buffers, ordered from oldest to newest updates.

  def _cache_reuse_buffer(self, rawbufs: Deque[Tuple[Any, int]]): # The newest cached buffer is reused.
    GlobalCounters.mem_cached -= self._underlying_buf_memsz(rawbufs[0][0])
    return rawbufs.popleft()[0]

  def ensure_has_free_space(self, space_to_free, device):
    while len(self.aging_order[device]) and self._get_cur_free_space(device) < space_to_free: # When OOM removing lru buffers.
      bucket, epoch = self.aging_order[device].popleft()
      if self.cached_buffers[bucket] and self.cached_buffers[bucket][-1][1] == epoch: self._free_buffer(self.cached_buffers[bucket].pop()[0]) # Free cached buffer if it is still in cache.
    assert (curr_free := self._get_cur_free_space(device)) > space_to_free, f"out of memory - requested: {space_to_free/1e9:5.2f} GB, available: {curr_free/1e9:5.2f} GB"

  def _alloc_buffer(self, size, dtype, device, **kwargs):
    self.ensure_has_free_space(size*dtype.itemsize, device)
    while True:
      try:
        newbuf = self._do_alloc(max(1, size), dtype, device, **kwargs)
        break
      except Exception:
        if len(self.aging_order[device]) == 0: raise
        self.ensure_has_free_space(1.1*self._get_cur_free_space(device), device) # increase free space by 10% and try again.
    self.free_space[device] -= size*dtype.itemsize
    self.buffer_info[newbuf] = (size, dtype, device)
    return newbuf

  def _free_buffer(self, buf_to_free):
    self.free_space[self.buffer_info[buf_to_free][2]] += self._underlying_buf_memsz(buf_to_free)
    GlobalCounters.mem_cached -= self._underlying_buf_memsz(buf_to_free)
    self.buffer_info.pop(buf_to_free)
    self._do_free(buf_to_free)

  def __call__(self, size, dtype, device='0', **kwargs): # allocate
    rawbufs = self.cached_buffers.get(self._cached_bufkey(size, dtype, device), None)
    return self._cache_reuse_buffer(rawbufs) if rawbufs else self._alloc_buffer(size, dtype, device, **kwargs)

  def free(self, buf): # free() just caches buffer. It might be freed later when OOM during allocation.
    self.epoch += 1
    size, dtype, device = self.buffer_info[buf]
    self.cached_buffers[self._cached_bufkey(size, dtype, device)].appendleft((buf, self.epoch))
    self.aging_order[device].append((self._cached_bufkey(size, dtype, device), self.epoch))
    GlobalCounters.mem_cached += self._underlying_buf_memsz(buf)

  def _underlying_buf_memsz(self, buf): return self.buffer_info[buf][0] * self.buffer_info[buf][1].itemsize
  def _cached_bufkey(self, size, dtype, device) -> Tuple[int, ...]: return (device, size, dtype, dtype.shape) if isinstance(dtype, ImageDType) else (device, size, dtype) # Provides a key for reusing device buffers with identical keys.
  def _do_alloc(self, size, dtype, device, **kwargs): raise NotImplementedError("must be implemented")
  def _do_free(self, buf): pass
  def _get_cur_free_space(self, device): return self.free_space[device]
