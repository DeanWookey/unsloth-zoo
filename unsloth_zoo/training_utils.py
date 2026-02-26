# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import math
import datasets
import threading
from transformers import set_seed as transformers_set_seed
from transformers import get_scheduler as transformers_get_scheduler
from transformers import Trainer
from transformers.trainer_utils import seed_worker as trainer_utils_seed_worker
from tqdm import tqdm as ProgressBar
import time
from typing import Any, Optional, List, Dict, Tuple
from .utils import _get_dtype, Version
from .hf_utils import dtype_from_config
from .gradient_checkpointing import (
    unpatch_unsloth_gradient_checkpointing,
    unpatch_unsloth_smart_gradient_checkpointing,
)
import os
import re

__all__ = [
    "fix_zero_training_loss",
    "unsloth_train",
    "prepare_model_for_training",
    "patch_trainer_for_memory_debugging",
    "patch_paged_optimizer_resume_fix",
]


@torch.inference_mode
def fix_zero_training_loss(model, tokenizer, train_dataset):
    """
    Sometimes the labels get masked by all -100s, causing the loss
    to be 0. We check for this!
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if isinstance(train_dataset, datasets.IterableDataset):
        # Skip the check since the code below assumes
        # an indexable dataset
        return
    
    if len(train_dataset) == 0: return


    row = train_dataset[0]
    if type(row) is dict and "labels" in row:

        # Check the first 100 rows
        seen_bad  = 0
        seen_good = 0
        for i, row in enumerate(train_dataset):
            try:    check_tokens = list(set(row["labels"]))
            except: continue
            if len(check_tokens) == 1 and check_tokens[0] == -100: seen_bad += 1
            else: seen_good += 1
            if i >= 100: break
        pass

        # Check ratio
        if seen_bad == 0 and seen_good == 0: return

        elif seen_bad / (seen_bad + seen_good) == 1:
            raise ZeroDivisionError(
                "Unsloth: All labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"\
                "If you used `train_on_responses_only`, confirm your user and assistant parts are correct!"
            )
        elif seen_bad / (seen_bad + seen_good) >= 0.9:
            print(
                "Unsloth: Nearly all labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"\
                "If you used `train_on_responses_only`, confirm your user and assistant parts are correct!"
            )
    pass
pass


def _mem(label):
    """Print current GPU memory allocated and reserved."""
    if not torch.cuda.is_available(): return
    alloc   = torch.cuda.memory_allocated()  / 1024**3
    reserved = torch.cuda.memory_reserved()  / 1024**3
    print(f"[MEM] {label:<55} alloc={alloc:.2f}GB  reserved={reserved:.2f}GB")
pass


@torch.no_grad
def prepare_model_for_training(
    model                      : Any,
    use_gradient_checkpointing : Optional = "unsloth",
    use_reentrant              : Optional[bool] = True,
    full_finetuning            : Optional[bool] = False,
    train_layernorms           : Optional[bool] = False,
    train_embedding            : Optional[bool] = False,
    train_lm_head              : Optional[bool] = False,
    float32_mixed_precision    : Optional[bool] = True,
    patch_modules_to_save      : Optional[bool] = False,
) -> Any:
    # All Unsloth Zoo code licensed under LGPLv3
    assert(use_gradient_checkpointing in (True, False, "unsloth",))
    assert(type(use_reentrant) is bool)
    assert(type(full_finetuning) is bool)
    assert(type(train_layernorms) is bool)
    assert(type(train_embedding) is bool)
    assert(type(train_lm_head) is bool)
    assert(type(float32_mixed_precision) is bool)

    print(f"\n{'='*70}")
    print(f"[DEBUG] prepare_model_for_training() called")
    print(f"[DEBUG]   full_finetuning         = {full_finetuning}")
    print(f"[DEBUG]   float32_mixed_precision = {float32_mixed_precision}")
    print(f"[DEBUG]   patch_modules_to_save   = {patch_modules_to_save}")
    print(f"[DEBUG]   use_gradient_checkpointing = {use_gradient_checkpointing}")
    _mem("START of prepare_model_for_training")

    dtype = _get_dtype(dtype_from_config(model.config))
    mixed_precision_dtype = torch.float32
    if dtype == torch.float16:
        # We need to upcast to float32
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
        # For full finetuning, update config dtype to match actual weight dtype.
        # The KV cache uses model.config.torch_dtype, but weights are upcast to float32.
        # Without this, generation fails with dtype mismatch in index_copy_().
        if full_finetuning:
            model._unsloth_original_dtype = dtype
            model.config.torch_dtype = torch.float32
    elif dtype == torch.bfloat16 and float32_mixed_precision:
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
        if full_finetuning:
            model._unsloth_original_dtype = dtype
            model.config.torch_dtype = torch.float32
    elif dtype == torch.bfloat16:
        mixed_precision_dtype = torch.bfloat16
        os.environ["UNSLOTH_MIXED_PRECISION"] = "bfloat16"
    else:
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
    pass

    print(f"[DEBUG]   model dtype             = {dtype}")
    print(f"[DEBUG]   mixed_precision_dtype   = {mixed_precision_dtype}")
    _mem("After dtype detection")

    # Count what we're about to upcast before doing it
    lora_params        = [(n, p) for n, p in model.named_parameters() if ".lora_A." in n or ".lora_B." in n or ".lora_magnitude_vector" in n]
    modules_save_params = [(n, p) for n, p in model.named_parameters() if "modules_to_save" in n]
    other_trainable    = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n not in dict(lora_params) and n not in dict(modules_save_params)]

    lora_numel  = sum(p.numel() for _, p in lora_params)
    mts_numel   = sum(p.numel() for _, p in modules_save_params)
    other_numel = sum(p.numel() for _, p in other_trainable)

    print(f"\n[DEBUG] Parameter inventory BEFORE upcast:")
    print(f"[DEBUG]   LoRA params (A/B):     {len(lora_params):>6} tensors  {lora_numel/1e6:>8.1f}M params  "
          f"fp32={lora_numel*4/1e9:.2f}GB  bf16={lora_numel*2/1e9:.2f}GB")
    print(f"[DEBUG]   modules_to_save:        {len(modules_save_params):>6} tensors  {mts_numel/1e6:>8.1f}M params  "
          f"fp32={mts_numel*4/1e9:.2f}GB  bf16={mts_numel*2/1e9:.2f}GB")
    print(f"[DEBUG]   other trainable:        {len(other_trainable):>6} tensors  {other_numel/1e6:>8.1f}M params")

    # Show dtype breakdown of LoRA params before upcast
    lora_dtype_counts = {}
    for _, p in lora_params:
        k = str(p.dtype)
        lora_dtype_counts[k] = lora_dtype_counts.get(k, 0) + 1
    print(f"[DEBUG]   LoRA dtype breakdown:  {lora_dtype_counts}")

    # Show dtype breakdown of modules_to_save before upcast
    if modules_save_params:
        mts_dtype_counts = {}
        for _, p in modules_save_params:
            k = str(p.dtype)
            mts_dtype_counts[k] = mts_dtype_counts.get(k, 0) + 1
        print(f"[DEBUG]   modules_to_save dtypes: {mts_dtype_counts}")

    for name, param in model.named_parameters():
        upcast = False
        requires_grad = False
        if not full_finetuning:
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                upcast = True
                requires_grad = True
            else:
                requires_grad = False
        else:
            if train_layernorms and ("norm." in name or "_layernorm" in name):
                requires_grad = True
                upcast = True # Must upcast layernorms to float32
            if train_embedding and ("embed_tokens" in name or "embedding" in name):
                requires_grad = True
                upcast = False # Can leave in bfloat16
            if train_lm_head and ("lm_head" in name):
                requires_grad = True
                upcast = False # Can leave in bfloat16
            else:
                requires_grad = True
                upcast = False # Can leave in bfloat16
        pass
        # Set training or not
        if requires_grad:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

        # Upcast to float32 if needed
        if requires_grad:
            name = name.replace("base_model", "model", 1)
            while re.search(r'\.(\d+)\.', name) is not None:
                name = re.sub(r'\.(\d+)\.', r'[\1].', name)
            name = name.replace(".weight", "", 1)
            dtype = torch.float32 if upcast else mixed_precision_dtype
            try:
                # Try original name
                exec(f"{name}.to({str(dtype)})")
            except:
                # Maybe model.model
                exec(f"model.{name}.to({str(dtype)})")
        pass

        if ('norm.' in name or '_layernorm' in name) and os.environ.get("UNSLOTH_UPCAST_LAYERNORM", "0") == "1":
            try:
                name = name.replace("base_model", "model", 1)
                while re.search(r'\.(\d+)\.', name) is not None:
                    name = re.sub(r'\.(\d+)\.', r'[\1].', name)
                name = name.replace(".weight", "", 1)
                # Try original name
                exec(f"{name}.to({str(torch.float32)})")
            except:
                # Maybe model.model
                exec(f"model.{name}.to({str(torch.float32)})")
    pass

    _mem("After LoRA upcast loop")

    # Show dtype breakdown of LoRA params after upcast
    lora_dtype_counts_after = {}
    for _, p in lora_params:
        k = str(p.dtype)
        lora_dtype_counts_after[k] = lora_dtype_counts_after.get(k, 0) + 1
    print(f"[DEBUG]   LoRA dtype breakdown AFTER upcast: {lora_dtype_counts_after}")

    if lora_numel > 0:
        expected_delta_gb = lora_numel * 2 / 1e9  # bf16->fp32 costs 2 extra bytes per param
        print(f"[DEBUG]   Expected memory increase from upcast: ~{expected_delta_gb:.2f}GB")

    # Gradient checkpointing
    # If the user requested vanilla GC (True/False), ensure any prior Unsloth patch is undone.
    if use_gradient_checkpointing != "unsloth":
        unpatch_unsloth_gradient_checkpointing()
        unpatch_unsloth_smart_gradient_checkpointing()
    m = model
    while hasattr(m, "model"):
        if use_gradient_checkpointing == "unsloth":
            m._offloaded_gradient_checkpointing = True
        if use_gradient_checkpointing == True and hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable()
        m = m.model
    pass
    if use_gradient_checkpointing == "unsloth":
        m._offloaded_gradient_checkpointing = True
    if use_gradient_checkpointing == True and hasattr(m, "gradient_checkpointing_enable"):
        m.gradient_checkpointing_enable()

    # Also set HF version manually to stop failures
    if hasattr(model, "_set_gradient_checkpointing"):
        if use_gradient_checkpointing in (True, "unsloth"):
            model._set_gradient_checkpointing()
        else:
            # Ensure checkpointing stays disabled if explicitly requested.
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = False

    _mem("After gradient checkpointing setup")

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    pass

    # Upcast modules_to_save
    _mem("Before modules_to_save handling")
    if patch_modules_to_save:
        try:
            from peft.utils import ModulesToSaveWrapper
        except:
            ModulesToSaveWrapper = None

        mts_count = 0
        for name, module in model.named_modules():
            if type(module) is ModulesToSaveWrapper or "ModulesToSave" in name:
                if getattr(module, "original_module", None) is not None:
                    module.original_module.requires_grad_(False)
                if getattr(module, "modules_to_save", None) is not None:
                    for saved_module in module.modules_to_save.modules():
                        if hasattr(saved_module, "weight"):
                            mts_count += 1
                            if saved_module.weight.dtype == torch.float16:
                                print(f"Unsloth: Upcasting `{name}` from float16 to float32 since it's in `modules_to_save`. Also allowing gradients.")
                                saved_module.to(torch.float32)
                                saved_module.requires_grad_(True)
                            else:
                                print(f"Unsloth: Allowing gradients for `{name}` since it's in `modules_to_save`.")
                                saved_module.requires_grad_(True)
                    pass
                pass
            pass
        pass
        print(f"[DEBUG]   modules_to_save processed: {mts_count} modules")
    else:
        # Even if patch_modules_to_save=False, check if they exist and report
        try:
            from peft.utils import ModulesToSaveWrapper
            found = [(n, m) for n, m in model.named_modules()
                     if type(m) is ModulesToSaveWrapper or "ModulesToSave" in n]
            if found:
                total_mts_params = 0
                for n, m in found:
                    if getattr(m, "modules_to_save", None) is not None:
                        for sm in m.modules_to_save.modules():
                            if hasattr(sm, "weight"):
                                total_mts_params += sm.weight.numel()
                print(f"[DEBUG]   WARNING: {len(found)} ModulesToSaveWrapper(s) found but patch_modules_to_save=False!")
                print(f"[DEBUG]   These hold {total_mts_params/1e6:.1f}M params "
                      f"(fp32={total_mts_params*4/1e9:.2f}GB  bf16={total_mts_params*2/1e9:.2f}GB)")
                print(f"[DEBUG]   Their requires_grad status will NOT be set here.")
        except Exception as e:
            print(f"[DEBUG]   Could not inspect ModulesToSaveWrapper: {e}")
    pass

    _mem("END of prepare_model_for_training")
    print(f"{'='*70}\n")

    return model
pass


def patch_trainer_for_memory_debugging(trainer):
    """
    Patches key HF Trainer methods to log GPU memory at every major step
    during trainer.train() startup. Call this BEFORE trainer.train().

    Intercepts:
      - create_optimizer
      - create_scheduler
      - _load_optimizer_and_scheduler  (resume path - where OOM is suspected)
      - _get_train_dataloader
      - training_step  (first step only, then unpatches itself)

    Also starts a background thread that logs memory every 2s so you can
    see the progression even inside opaque C++/CUDA calls.
    """
    # Background memory monitor thread
    _stop_monitor = threading.Event()
    _last_reported = [torch.cuda.memory_allocated() / 1024**3]

    def _monitor_loop():
        while not _stop_monitor.is_set():
            try:
                alloc    = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved()  / 1024**3
            except Exception:
                break  # CUDA shutting down, exit cleanly
            delta = alloc - _last_reported[0]
            if abs(delta) >= 0.5:  # Only print when >= 0.5 GB change
                print(f"[MEMMON] alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  delta={delta:+.2f}GB")
                _last_reported[0] = alloc
            time.sleep(0.5)

    monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
    monitor_thread.start()
    print(f"[DEBUG] Memory monitor started (reports on >=0.5GB changes)")
    _mem("patch_trainer_for_memory_debugging - INITIAL")

    # ------------------------------------------------------------------ #
    # Helper: wrap a bound method with before/after memory prints
    # ------------------------------------------------------------------ #
    def _wrap(obj, method_name, label):
        original = getattr(obj, method_name, None)
        if original is None:
            print(f"[DEBUG] WARNING: {method_name} not found on trainer - skipping patch")
            return

        def _wrapped(*args, **kwargs):
            _mem(f"BEFORE {label}")
            result = original(*args, **kwargs)
            _mem(f"AFTER  {label}")
            return result

        import types
        setattr(obj, method_name, types.MethodType(
            lambda self, *a, **kw: _wrapped(*a, **kw), obj
        ))

    # ------------------------------------------------------------------ #
    # Patch create_optimizer
    # ------------------------------------------------------------------ #
    _orig_create_optimizer = trainer.create_optimizer.__func__ \
        if hasattr(trainer.create_optimizer, '__func__') else None

    _orig_create_optimizer_bound = trainer.create_optimizer

    def _patched_create_optimizer():
        _mem("BEFORE create_optimizer")
        result = _orig_create_optimizer_bound()
        _mem("AFTER  create_optimizer")
        # Print optimizer param group sizes
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            for i, pg in enumerate(trainer.optimizer.param_groups):
                n_params = sum(p.numel() for p in pg['params'])
                print(f"[DEBUG]   optimizer param_group[{i}]: {n_params/1e6:.1f}M params")
        return result

    import types
    trainer.create_optimizer = _patched_create_optimizer

    # ------------------------------------------------------------------ #
    # Patch _load_optimizer_and_scheduler (resume-specific - key suspect)
    # ------------------------------------------------------------------ #
    if hasattr(trainer, '_load_optimizer_and_scheduler'):
        _orig_load_opt_sched = trainer._load_optimizer_and_scheduler

        def _patched_load_opt_sched(resume_from_checkpoint):
            print(f"\n[DEBUG] _load_optimizer_and_scheduler called with: {resume_from_checkpoint}")
            _mem("BEFORE _load_optimizer_and_scheduler")

            # Check what files exist in the checkpoint
            if resume_from_checkpoint and isinstance(resume_from_checkpoint, str):
                import os
                opt_files = [f for f in os.listdir(resume_from_checkpoint)
                             if 'optimizer' in f.lower() or 'scheduler' in f.lower()]
                for f in opt_files:
                    size_gb = os.path.getsize(
                        os.path.join(resume_from_checkpoint, f)
                    ) / 1024**3
                    print(f"[DEBUG]   checkpoint file: {f}  size={size_gb:.2f}GB")

            result = _orig_load_opt_sched(resume_from_checkpoint)
            _mem("AFTER  _load_optimizer_and_scheduler")
            return result

        trainer._load_optimizer_and_scheduler = _patched_load_opt_sched
    else:
        print("[DEBUG] WARNING: _load_optimizer_and_scheduler not found on trainer")

    # ------------------------------------------------------------------ #
    # Patch _get_train_dataloader
    # ------------------------------------------------------------------ #
    if hasattr(trainer, 'get_train_dataloader'):
        _orig_get_dataloader = trainer.get_train_dataloader

        def _patched_get_dataloader():
            _mem("BEFORE get_train_dataloader")
            result = _orig_get_dataloader()
            _mem("AFTER  get_train_dataloader")
            return result

        trainer.get_train_dataloader = _patched_get_dataloader

    # ------------------------------------------------------------------ #
    # Patch training_step to catch first step and stop monitor after crash
    # ------------------------------------------------------------------ #
    if hasattr(trainer, 'training_step'):
        _orig_training_step = trainer.training_step
        _step_count = [0]

        def _patched_training_step(model, inputs, *args, **kwargs):
            if _step_count[0] == 0:
                _mem("BEFORE first training_step (forward+backward)")
            try:
                result = _orig_training_step(model, inputs, *args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                _mem("OOM INSIDE training_step")
                _stop_monitor.set()
                raise
            if _step_count[0] == 0:
                _mem("AFTER  first training_step")
            _step_count[0] += 1
            return result

        trainer.training_step = _patched_training_step

    print(f"[DEBUG] Trainer patched. Call trainer.train() now.")
    print(f"[DEBUG] To stop memory monitor: call the returned stop_fn()")
    return _stop_monitor.set  # Return a function to stop the monitor


def patch_paged_optimizer_resume_fix(trainer):
    """
    Fixes OOM when resuming from checkpoint with paged optimizers (e.g. paged_adamw_8bit).

    Root cause:
        torch.save/load does NOT preserve the is_paged=True attribute on tensors.
        When optimizer states are loaded from a checkpoint, they land on GPU memory
        without is_paged=True. bitsandbytes normally pages optimizer states between
        CPU and GPU (only on GPU during the optimizer.step() call), but after
        checkpoint loading they stay on GPU permanently, wasting ~9-10GB.

    Fix:
        Wrap _load_optimizer_and_scheduler so that after loading, any GPU optimizer
        state tensors are moved back to CPU in-place and marked is_paged=True.
        bitsandbytes' prefetch_state() will then manage them correctly (it does
        A.data = A.data.cuda() / A.data = A.data.cpu() as needed during step()).

    Usage:
        Call this BEFORE trainer.train() when resuming from a checkpoint:

            from unsloth_zoo.training_utils import patch_paged_optimizer_resume_fix
            patch_paged_optimizer_resume_fix(trainer)
            trainer.train(resume_from_checkpoint=checkpoint_path)
    """
    if not hasattr(trainer, '_load_optimizer_and_scheduler'):
        print("[Unsloth] WARNING: _load_optimizer_and_scheduler not found on trainer, "
              "cannot apply paged optimizer resume fix")
        return

    _orig_load = trainer._load_optimizer_and_scheduler

    def _patched_load(resume_from_checkpoint):
        result = _orig_load(resume_from_checkpoint)

        # Only act when actually resuming from a checkpoint
        if resume_from_checkpoint is None:
            return result

        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is None:
            print("[Unsloth] Paged optimizer fix: trainer.optimizer is None after load")
            return result

        # HF Trainer wraps the optimizer in AcceleratedOptimizer (or similar).
        # Unwrap to find the actual bitsandbytes optimizer for the is_paged check.
        inner = optimizer
        while hasattr(inner, 'optimizer') and inner is not getattr(inner, 'optimizer', None):
            inner = inner.optimizer

        inner_module = getattr(type(inner), '__module__', '') or ''
        inner_name   = type(inner).__name__
        outer_paged  = getattr(optimizer, 'is_paged', None)
        inner_paged  = getattr(inner,     'is_paged', None)
        is_bnb       = 'bitsandbytes' in inner_module
        print(f"[Unsloth] Optimizer: {type(optimizer).__name__} / inner: {inner_name} ({inner_module})")
        print(f"[Unsloth] is_paged: outer={outer_paged}, inner={inner_paged}, is_bnb={is_bnb}")
        print(f"[Unsloth] State sizes: outer={len(optimizer.state)}, inner={len(inner.state)}")

        # Apply fix to:
        #   a) explicitly paged optimizers (PagedAdamW8bit)  — is_paged=True on inner/outer
        #   b) any other bitsandbytes 8-bit optimizer (AdamW8bit etc.) — detected via module
        # For both cases we move GPU states to CPU and enable bitsandbytes paging
        # so states are only on GPU transiently during optimizer.step().
        # For plain PyTorch optimizers we skip (states must stay on GPU).
        already_paged = bool(outer_paged) or bool(inner_paged)
        if not (already_paged or is_bnb):
            print("[Unsloth] Paged optimizer fix: not a bitsandbytes optimizer, skipping")
            return result

        # bitsandbytes paging works by marking state tensors with is_paged=True
        # and page_deviceid, then calling prefetch_state/unget_state in step().
        # torch.load loses these attributes so loaded states stay on GPU.
        #
        # Problem: bitsandbytes' prefetch_tensor() calls cudaMemPrefetchAsync,
        # which only works on CUDA Unified Memory (cudaMallocManaged). Our tensors
        # moved via v.data.cpu() are plain CPU tensors, not managed memory, so
        # cudaMemPrefetchAsync returns "invalid argument" and crashes the kernel.
        #
        # Fix: patch bitsandbytes.functional.prefetch_tensor to use Python-level
        # tensor moves (A.data = A.data.cuda() / A.data = A.data.cpu()) instead
        # of the CUDA API. This is correct for all tensor types.
        try:
            import bitsandbytes.functional as bnb_f
            _orig_prefetch = bnb_f.prefetch_tensor

            def _python_prefetch(A, to_cpu=False):
                if to_cpu:
                    if A.data.device.type != 'cpu':
                        A.data = A.data.cpu()
                else:
                    if A.data.device.type == 'cpu':
                        A.data = A.data.cuda(getattr(A, 'page_deviceid', 0))

            bnb_f.prefetch_tensor = _python_prefetch
            print("[Unsloth] Patched bitsandbytes.functional.prefetch_tensor "
                  "to use Python-level tensor moves (avoids cudaMemPrefetchAsync "
                  "on non-managed memory)")
        except Exception as e:
            print(f"[Unsloth] Could not patch prefetch_tensor: {e}")

        device_id = torch.cuda.current_device()
        moved = 0
        freed_bytes = 0
        for group in inner.param_groups:
            for p in group['params']:
                if p not in inner.state:
                    continue
                for k, v in inner.state[p].items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        freed_bytes += v.numel() * v.element_size()
                        v.data = v.data.cpu()        # in-place: same Python object, CPU storage
                        v.is_paged = True             # mark for bitsandbytes prefetch_state check
                        v.page_deviceid = device_id   # device index (needed by prefetch_tensor)
                        moved += 1

        if moved > 0:
            if not getattr(inner, 'is_paged', False):
                inner.is_paged = True  # enable paging on optimizer if not already set
                print(f"[Unsloth] Enabled paging on {inner_name} optimizer")
            torch.cuda.empty_cache()
            print(f"[Unsloth] Paged optimizer fix: moved {moved} state tensors "
                  f"to CPU, freed ~{freed_bytes/1e9:.2f}GB GPU memory")
        else:
            print(f"[Unsloth] Paged optimizer fix: no GPU state tensors found "
                  f"(inner state count={len(inner.state)})")

        return result

    trainer._load_optimizer_and_scheduler = _patched_load
    print("[Unsloth] Paged optimizer resume fix installed. "
          "Optimizer states will be moved to CPU after checkpoint loading.")


def get_max_steps(training_args, n_training_samples, train_dataset):
    # Approximately from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2092
    # Determines batch size, max steps, ga etc
    if training_args.world_size > 1:
        raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
    pass

    bsz = training_args.per_device_train_batch_size
    ga  = training_args.gradient_accumulation_steps

    total_train_batch_size = bsz * ga
    max_steps = training_args.max_steps

    if max_steps > 0:
        total_samples_seen = total_train_batch_size * max_steps
        num_train_epochs = math.ceil(total_samples_seen / n_training_samples)
    else:
        num_train_epochs = training_args.num_train_epochs
        steps_per_epoch  = math.ceil(n_training_samples / total_train_batch_size)
        max_steps = math.ceil(steps_per_epoch * num_train_epochs)
        num_train_epochs = math.ceil(num_train_epochs)
    return total_train_batch_size, max_steps, num_train_epochs
pass


def set_training(model):
    # Start training
    model.training = True
    while hasattr(model, "model"):
        model = model.model
        model.training = True
    model.training = True
pass


def unset_training(model):
    # End training
    model.training = False
    while hasattr(model, "model"):
        model = model.model
        model.training = False
    model.training = False
pass


from dataclasses import dataclass
@dataclass
class Trainer_Stats:
    metrics: dict
pass

def unsloth_train(trainer):
    """
    Unsloth Trainer
    1. Fixes gradient accumulation
    2. Scaled down version of HF's trainer
    3. Much less feature complete
    """
    # All Unsloth Zoo code licensed under LGPLv3
    assert(hasattr(trainer, "args"))
    assert(hasattr(trainer, "model"))
    assert(hasattr(trainer, "train_dataset"))
    assert(hasattr(trainer, "data_collator"))

    model = trainer.model
    training_args = trainer.args
    data_collator = trainer.data_collator
    n_training_samples = len(trainer.train_dataset)
    set_training(model)
    transformers_set_seed(training_args.seed)

    if training_args.dataloader_drop_last:
        raise NotImplementedError(
            "Unsloth: Currently `dataloader_drop_last` is not yet implemented!"
        )
    pass

    if data_collator is None:
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = trainer.tokenizer,
            mlm = False,
            pad_to_multiple_of = 4,
        )
    pass

    # Separate weight decay for parameters
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    decay_parameters = frozenset(Trainer.get_decay_parameter_names(None, model))
    yes_decay, no_decay = [], []
    n_parameters_to_train = 0
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name in decay_parameters: yes_decay.append(param)
        else: no_decay.append(param)
        n_parameters_to_train += param.numel()
    pass
    optimizer_grouped_parameters = [
        {"params" : yes_decay, "weight_decay" : training_args.weight_decay,},
        {"params" : no_decay,  "weight_decay" : 0,}
    ]
    trainable_parameters = \
        optimizer_grouped_parameters[0]["params"] + \
        optimizer_grouped_parameters[1]["params"]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    total_train_batch_size, max_steps, num_train_epochs = \
        get_max_steps(training_args, n_training_samples, trainer.train_dataset)

    # Get LR scheduler
    lr_scheduler = transformers_get_scheduler(
        name = training_args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps = training_args.get_warmup_steps(max_steps),
        num_training_steps = max_steps,
        **getattr(training_args, "lr_scheduler_kwargs", {}),
    )

    # Gradient accumulation and grad norm clipping
    max_grad_norm   = training_args.max_grad_norm
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    bsz = training_args.per_device_train_batch_size
    ga  = training_args.gradient_accumulation_steps
    # inverse_gradient_accumulation_steps = 1.0 / ga
    # inverse_gradient_accumulation_steps = \
    #     torch.FloatTensor([inverse_gradient_accumulation_steps])\
    #     .to(device = "cuda:0", non_blocking = True)[0]

    # Mixed precision scaling
    torch_version = torch.__version__
    config_dtype = dtype_from_config(model.config)
    if config_dtype == torch.float16:
        mixed_precision = "fp16"
        mixed_dtype = torch.float16
        # torch.cuda.amp.autocast is deprecated >= 2.4
        if Version(torch_version) < Version("2.4.0"):
            float16_scaler = torch.cuda.amp.GradScaler()
        else:
            float16_scaler = torch.amp.GradScaler("cuda")
    else:
        mixed_precision = "bf16"
        mixed_dtype = torch.bfloat16
        float16_scaler = None
    pass
    
    optimizer.zero_grad()

    # torch.cuda.amp.autocast is deprecated >= 2.4
    torch_version = torch.__version__
    if Version(torch_version) < Version("2.4.0"):
        autocast_context_manager = torch.cuda.amp.autocast(
            dtype = mixed_dtype,
            cache_enabled = False,
        )
    else:
        autocast_context_manager = torch.amp.autocast(
            device_type = "cuda",
            dtype = mixed_dtype,
            cache_enabled = False,
        )
    pass

    step = 0
    accumulated_loss = torch.zeros(1, device = "cuda:0", dtype = torch.float32)[0]
    debug_info = \
        f'==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = {training_args.world_size}\n'\
        f'    \\   /|    Num examples = {n_training_samples:,} | Num Epochs = {num_train_epochs:,}\n'\
        f'O^O/ \\_/ \\    Batch size per device = {training_args.per_device_train_batch_size:,} | Gradient Accumulation steps = {training_args.gradient_accumulation_steps}\n'\
        f'\\        /    Total batch size = {total_train_batch_size:,} | Total steps = {max_steps:,}\n'\
        f' "-____-"     Number of trainable parameters = {n_parameters_to_train:,}'
    print(debug_info)

    # Get per epoch counter
    max_iters_per_epoch = math.ceil(n_training_samples / total_train_batch_size)
    leftover_samples = n_training_samples % total_train_batch_size
    # But also consider leftover steps
    leftover_ga = math.ceil(leftover_samples / bsz)
    if leftover_samples == 0: leftover_ga = ga

    logging_steps = training_args.logging_steps
    # Go through each epoch
    start_time = time.time()
    with ProgressBar(total = max_steps, dynamic_ncols = True) as progress_bar:
        for epoch in range(num_train_epochs):

            # We also need to shuffle the data loader every epoch!
            transformers_set_seed(training_args.seed + epoch)
            train_dataloader_iterator = iter(torch.utils.data.DataLoader(
                trainer.train_dataset,
                batch_size     = bsz,
                sampler        = torch.utils.data.SequentialSampler(trainer.train_dataset),
                num_workers    = training_args.dataloader_num_workers,
                collate_fn     = data_collator,
                pin_memory     = training_args.dataloader_pin_memory,
                drop_last      = training_args.dataloader_drop_last,
                worker_init_fn = trainer_utils_seed_worker,
            ))

            for j in range(max_iters_per_epoch):
                n_batches = leftover_ga if j == (max_iters_per_epoch-1) else ga
                batches = [next(train_dataloader_iterator) for j in range(n_batches)]

                # Count non zeros before loss calc
                n_items = torch.stack([
                    torch.count_nonzero(x["labels"][..., 1:] != -100) for x in batches
                ]).sum()

                # Gradient accumulation
                for batch in batches:
                    input_ids = batch["input_ids"].pin_memory().to(device = "cuda:0", non_blocking = True)
                    labels    = batch["labels"]   .pin_memory().to(device = "cuda:0", non_blocking = True)

                    with autocast_context_manager:
                        loss = model(input_ids = input_ids, labels = labels, n_items = n_items).loss
                        # loss = loss * inverse_gradient_accumulation_steps
                        accumulated_loss += loss.detach()
                    pass

                    if float16_scaler is None:  loss.backward()
                    else: float16_scaler.scale(loss).backward()
                pass

                if float16_scaler is None:
                    clip_grad_norm_(trainable_parameters, max_grad_norm)
                    optimizer.step()
                else:
                    float16_scaler.unscale_(optimizer)
                    clip_grad_norm_(trainable_parameters, max_grad_norm)
                    float16_scaler.step(optimizer)
                    float16_scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % logging_steps == 0:
                    progress_bar.write(f"{step}, {round(accumulated_loss.cpu().item(), 4)}")
                pass
                accumulated_loss.zero_()
                progress_bar.update(1)

                step += 1
                if step == max_steps: break
            pass
        pass
    pass
    unset_training(model)
    print("Unsloth: Finished training!")
    end_time = time.time()

    # Return stats
    trainer_stats = Trainer_Stats(metrics = {"train_runtime" : end_time - start_time})
    return trainer_stats
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
