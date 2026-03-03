import os
import warnings
from contextlib import nullcontext
from typing import Literal, cast

import accelerate
import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel
from torch.autograd import profiler
from torch.nn.modules.module import _IncompatibleKeys
from tqdm import tqdm

from ..config_utils import function_config_to_basic_types
from loguru import logger


@profiler.record_function("load_weights_with_shape_check")
def load_weights_with_shape_check(
    module: nn.Module,
    weights: dict,
    load_strategy: Literal["pair", "search"] = "search",
) -> _IncompatibleKeys:
    """
    Load weights into the module with shape checking and return missing/unexpected keys.

    Args:
        module: The target module to load weights into
        weights: Dictionary of weights to load (name -> tensor)

    Returns:
        Tuple containing:
        - missing_keys: List of parameter names not found in weights
        - unexpected_keys: List of weight names not found in module
    """
    # if model has some rigional compiled modules, unwrap the model first and then load the weight
    # may meet the __dict__['_orig_mod'] bug, see merge: https://github.com/huggingface/accelerate/pull/3881
    # make sure install the leatest accelerate package
    module = accelerate.utils.extract_model_from_parallel(module)

    missing_keys = []
    unexpected_keys = list(weights.keys())  # Start with all keys, remove matched ones

    # faster but need paire checkpoints, suitable for the model weights are with most same keys
    # but only few keys are different
    if load_strategy == "pair":
        param_items = list(module.named_parameters())
        weight_items = list(weights.items())

        if len(param_items) != len(weight_items):
            logger.warning(f"Parameter count mismatch: model has {len(param_items)}, weights have {len(weight_items)}")

        for (name, param), (weight_name, weight) in zip(param_items, weight_items):
            if name != weight_name:
                logger.warning(f"Name mismatch: model has '{name}', weights have '{weight_name}'")
                missing_keys.append(name)
                if weight_name in unexpected_keys:
                    unexpected_keys.remove(weight_name)
                continue
            if param.shape == weight.shape:
                param.data.copy_(weight.data)
                unexpected_keys.remove(weight_name)
                logger.debug(f"Weights loaded for {name} (shape: {param.shape})")
            else:
                logger.warning(
                    f"Shape mismatch for {name}: expected {param.shape}, got {weight.shape} - skipping",
                )
                missing_keys.append(name)
                unexpected_keys.remove(weight_name)

    elif load_strategy == "search":
        params = list(module.named_parameters())
        tbar = tqdm(params, desc="Loading model checkpoint ...")
        for name, param in tbar:
            tbar.set_description(f"Meteralizing param={name}")
            if name in unexpected_keys:  # search in the whole weight keys, O(n) complexity
                unexpected_keys.remove(name)  # This key was expected, and remove from the total key set
                if param.shape == weights[name].shape:
                    param.data.copy_(weights[name].data)
                else:
                    logger.warning(
                        f"Shape mismatch for {name}: expected {param.shape}, got {weights[name].shape} - skipping",
                        tqdm=True,
                    )
                    # Consider this a missing key since we didn't load it
                    missing_keys.append(name)
            else:
                missing_keys.append(name)
                logger.warning(f"{name} not found in weights", tqdm=True)

    else:
        raise ValueError(f"Invalid load strategy: {load_strategy}")

    return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)


def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
    """
    Load a Fully Sharded Data Parallel (FSDP) model's state dictionary from disk.

    This function supports loading models saved in different formats including full,
    local, and sharded state dictionaries. Handles special cases for distributed
    environments and adapter-only loading modes.

    Parameters:
        fsdp_plugin: FSDPPlugin object containing configuration parameters
        accelerator: Accelerator instance for distributed environment management
        model: The target model to load weights into (may be FSDP-wrapped)
        input_dir: Directory containing the model checkpoint files
        model_index: Index of the model to load when multiple models exist (default: 0)
        adapter_only: Whether to load only adapter weights (LoRA/PEFT) (default: False)

    Returns:
        Any: Result from the underlying state dict loading operation, typically
             an object containing information about missing/unexpected keys
    """

    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from accelerate.utils.constants import (
        FSDP_MODEL_NAME,
    )
    from accelerate.utils.fsdp_utils import _get_model_state_dict, _set_model_state_dict
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT and fsdp_plugin.fsdp_version == 1:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model,
            fsdp_plugin.state_dict_type,
            fsdp_plugin.state_dict_config,
            fsdp_plugin.optim_state_dict_config,
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )

    with ctx:
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) is not FSDP and accelerator.process_index != 0:
                if not fsdp_plugin.sync_module_states and fsdp_plugin.fsdp_version == 1:
                    raise ValueError(
                        "Set the `sync_module_states` flag to `True` so that model states are synced across processes when "
                        "initializing FSDP object"
                    )
                return
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = (
                os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{model_index}")
                if f"{FSDP_MODEL_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading model from {ckpt_dir}")
            state_dict = {"model": _get_model_state_dict(model, adapter_only=adapter_only)}
            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
                planner=DefaultLoadPlanner(),
            )
            state_dict = state_dict["model"]
            logger.info(f"Model loaded from {ckpt_dir}")

        if fsdp_plugin.fsdp_version == 1:
            load_result = _set_model_state_dict(model, state_dict, adapter_only=adapter_only)
        else:
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            # not all keys are loaded by dist_cp because of adapter_only=True
            # so we need to load the rest of the keys
            if adapter_only:
                state_dict = remap_peft_model_state_dict(model, state_dict)[0]
                _model_sd = _get_model_state_dict(model, False)
                for key, weight in state_dict.items():  # 96
                    assert key in _model_sd, f"{key} not in model state dict"
                    _model_sd[key] = weight  # inplace copy -> 390
                state_dict = _model_sd

            load_result = set_model_state_dict(model, state_dict)
    return load_result


def remap_peft_model_state_dict(
    model,
    peft_model_state_dict,
    adapter_name="default",
    ignore_mismatched_sizes: bool = False,
):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]):
            The Peft model.
        peft_model_state_dict (`dict`):
            The state dict of the Peft model.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be set.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether to ignore mismatched in the state dict.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            This argument must be `True` if the `model` was loaded with adapter weights on the meta device, e.g. after
            calling `inject_adapter_in_model` with `low_cpu_mem_usage=True`. Otherwise, leave it as `False`.

    """
    import warnings

    from peft import PeftType
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils.other import AuxiliaryTrainingWrapper
    from peft.utils.save_and_load import (
        _find_mismatched_keys,
        _insert_adapter_name_into_state_dict,
    )

    config = model.peft_config[adapter_name]
    state_dict = peft_model_state_dict

    # handle auxiliary training wrappers such as ModulesToSaveWrapper and TrainableTokensWrapper by getting each of
    # them and translating saved state dict key (which does not include the adapter name) to loaded state dict key
    # (which includes the adapter name).
    for name, module in model.named_modules():
        if isinstance(module, AuxiliaryTrainingWrapper):
            # Not every module has a 1:1 mapping. ModulesToSaveWrapper, for example, removes the
            # `modules_to_save.{adapter_name}.` prefix. This prefix must be restored when loading the model from the
            # saved state dict which is why we fetch a load key map from the wrapper.
            key_map = module.adapter_state_dict_load_map(adapter_name)
            for k in key_map:
                lookup_key = f"{name}.{k}"
                store_key = f"{name}.{key_map[k]}"

                state_dict[store_key] = peft_model_state_dict[lookup_key]

                # delete the old key from the previous `state_dict = peft_model_state_dict` statement.
                del state_dict[lookup_key]

    if config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    elif config.peft_type == PeftType.XLORA:
        peft_model_state_dict = state_dict
    elif config.peft_type in PEFT_TYPE_TO_PREFIX_MAPPING:
        peft_model_state_dict = {}
        parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        if config.peft_type == PeftType.VBLORA and config.save_only_topk_weights:
            num_vectors, _ = model.vblora_vector_bank[adapter_name].shape
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                # in save_only_topk_weights mode, only topk_indices and topk_weights are saved
                # note that topk_indices and topk_weights serve as an efficient representation of the logits
                # so we need to recover the logits from the topk_indices and topk_weights
                if "_topk_indices" in k:
                    v = state_dict[k].to(torch.long)
                    original_key = k.replace("_topk_indices", "")
                    # find the corresponding topk_weights from the state_dict
                    topk_weights = state_dict[k.replace("_topk_indices", "_topk_weights")]
                    # as we only save the first k-1 topk_weights, here we recover the last one
                    topk_weights = torch.cat([topk_weights, 1 - topk_weights.sum(-1, keepdim=True)], dim=-1)
                    # convert the weights to logits
                    topk_logits = torch.log(topk_weights)
                    matrix = (
                        torch.zeros([*(topk_logits.shape[:-1]), num_vectors])
                        .fill_(float("-inf"))
                        .to(topk_logits.device)
                        .scatter(-1, v, topk_logits)
                    )
                    # add logits to the state_dict
                    state_dict[original_key] = matrix
                    # delete the topk_indices and topk_weights from the state_dict
                    del state_dict[k]
                    del state_dict[k.replace("_topk_indices", "_topk_weights")]

        peft_model_state_dict = _insert_adapter_name_into_state_dict(
            state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
        )

        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
        elif config.peft_type == PeftType.VERA:
            if config.save_projection and "base_model.vera_A" not in peft_model_state_dict:
                raise ValueError(
                    "Specified to load vera_A and vera_B from state dictionary however they were not present!"
                )
            elif not config.save_projection and "base_model.vera_A" in peft_model_state_dict:
                warnings.warn(
                    "Specified to not load vera_A and vera_B from state dictionary however they are present in state"
                    " dictionary! Consider using them to ensure checkpoint loading is correct on all platforms using"
                    " `peft_config.save_projection = True`"
                )
            elif not config.save_projection:  # and no vera_A in state dictionary
                warnings.warn(
                    "Specified to not load vera_A and vera_B from state dictionary. This means we will be relying on"
                    " PRNG initialisation to restore these projections using `config.projection_prng_key`, which may"
                    " not be accurate on all system configurations."
                )
        elif config.peft_type == PeftType.LORA:
            # Here we take care of a refactor of DoRA which changed lora_magnitude_vector from a ParameterDict to a
            # ModuleDict with a DoraLayer instance. The old parameter is now the "weight" attribute of that layer.
            old_dora_suffix = f"lora_magnitude_vector.{adapter_name}"

            def renamed_dora_weights(k):
                if k.endswith(old_dora_suffix):
                    k = k + ".weight"
                return k

            peft_model_state_dict = {renamed_dora_weights(k): v for k, v in peft_model_state_dict.items()}
    else:
        raise NotImplementedError

    peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
        model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    )

    return peft_model_state_dict, mismatched_keys


def load_peft_model_checkpoint(
    base_model: nn.Module,
    peft_pretrained_path: str,
    merge_and_unload: bool = True,
    base_model_pretrained_path: str | None = None,
) -> tuple[PeftConfig, PeftModel | nn.Module]:
    """
    Load a PEFT model checkpoint and merge it with the base model.

    Args:odel (nn.Module): The base model to merge with.
        base_model_pretrained_path (str | None): Path to the base model checkpoint.
        peft_pretrained_path (str): Path to the PEFT model checkpoint.
        merge_and_unload (bool): Whether to merge and unload the PEFT model.

    Returns:
        base_model: The base model after loading the checkpoint.
        tuple: A tuple containing the PeftConfig and the merged PeftModel or nn.Module.
    """
    if base_model_pretrained_path:
        base_sd = accelerate.utils.load_state_dict(base_model_pretrained_path)
        # Load the base model
        _incompact_keys = base_model.load_state_dict(base_sd, strict=False)
        logger.info(f"Base model loaded with incompatible keys:\n{_incompact_keys}")
    else:
        warnings.warn("No base model checkpoint provided. Please make sure the base model is initialized correctly.")

    # Load the PEFT model
    peft_config = PeftConfig.from_pretrained(peft_pretrained_path)
    peft_model = PeftModel.from_pretrained(base_model, peft_pretrained_path, adapter_name="default")

    if merge_and_unload:
        peft_model = peft_model.merge_and_unload(  # type: ignore
            progressbar=True, adapter_names=["default"]
        )

    return peft_config, peft_model


# Tokenizer lora module utilities


@function_config_to_basic_types
def load_diffbands_tokenizer_then_peft_lora(
    model_cls: nn.Module,
    tokenizer_pretrained_path: str,
    peft_pretrained_path: str,
    load_main_model_in_peft_fn: bool = False,  # if False, handling loading main checkpoint in the tokenizer class
    model_kwargs: dict | None = None,
    peft_kwargs: dict | None = None,
) -> tuple[PeftConfig, nn.Module | PeftModel]:
    """
    Loads a tokenizer model with a pretrained PEFT (Parameter-Efficient Fine-Tuning) configuration.
    This function initializes a tokenizer model using the provided model class and arguments,
    then loads a PEFT model checkpoint into the tokenizer. Optionally, it can also load the
    main model checkpoint during the PEFT loading process.
    Args:
        model_cls (nn.Module): The class of the tokenizer model to be instantiated.
        tokenizer_pretrained_path (str): Path to the pretrained tokenizer checkpoint.
        peft_pretrained_path (str): Path to the pretrained PEFT checkpoint.
        load_main_model_in_peft_fn (bool, optional): If True, the main model checkpoint is
            loaded during the PEFT loading process. If False, the main model checkpoint
            should be handled by the tokenizer class. Defaults to False.
        model_kwargs (dict | None): Keyword arguments to initialize the tokenizer model.
            Must be provided.
        peft_kwargs (dict | None): Additional keyword arguments for loading the PEFT model.
            Defaults to None.
    Returns:
        tuple[PeftConfig, nn.Module | PeftModel]: A tuple containing the PEFT configuration
        and the tokenizer model with the PEFT checkpoint loaded.
    Raises:
        AssertionError: If `model_kwargs` is not provided.
    Example:
        peft_config, tokenizer = load_diffbands_tokenizer_then_peft_lora(
            model_cls=MyTokenizerClass,
            tokenizer_pretrained_path="path/to/tokenizer",
            peft_pretrained_path="path/to/peft",
            load_main_model_in_peft_fn=True,
            model_kwargs={"arg1": value1, "arg2": value2},
            peft_kwargs={"peft_arg1": peft_value1}
    """
    if peft_kwargs is None:
        peft_kwargs = {}
    assert model_kwargs is not None, "model_kwargs must be provided"

    tokenizer = model_cls(**model_kwargs)

    peft_config, tokenizer = load_peft_model_checkpoint(
        tokenizer,
        peft_pretrained_path,
        base_model_pretrained_path=tokenizer_pretrained_path if load_main_model_in_peft_fn else None,
        **peft_kwargs,
    )

    logger.info(f"Tokenizer loaded with PEFT config: {peft_config}")

    return peft_config, tokenizer


def safe_init_weights(init_func):
    def _wrapper(*args, **kwargs):
        module = args[0]
        assert isinstance(module, nn.Module), f"{module} is not a nn.Module, it is a {type(module)}"

        # Check if module is on meta device
        params_iter = module.parameters()
        p = None
        try:
            p = next(params_iter)
        except StopIteration:
            pass

        if p is None:
            logger.info(f"Module {type(module)} got no params, skipping init")
            return

        # if using with torch.device('meta') to init, this will skip weight init,
        # use it when only try to load weight (not training at all).
        elif str(p.device) == "meta":
            logger.debug(f"Module params are on meta device, skipping weight initialization")
            return

        # Init function
        r = init_func(*args, **kwargs)

        return r

    return _wrapper


def get_unwrapped_state_dict(model: nn.Module):
    model: nn.Module = accelerate.utils.extract_model_from_parallel(model)
    state_dict = model.state_dict()
    # keep ordered
    state_dict_clear = {}
    for k in state_dict.keys():
        if "_orig_mod" in k:  # compiled submodule
            state_dict_clear[k.replace("_orig_mod", "")] = state_dict[k]
        else:
            state_dict_clear[k] = state_dict[k]
    return state_dict_clear


def unwrap_model_recursive(
    model: nn.Module | torch._dynamo.OptimizedModule,
    keep_submodule_compiled: bool = False,
) -> nn.Module | torch._dynamo.OptimizedModule:
    model = accelerate.utils.extract_model_from_parallel(
        model,
        keep_torch_compile=keep_submodule_compiled,
        recursive=True,
    )
    if keep_submodule_compiled:
        return model

    return _unwrap_torch_compile_submodules(cast(nn.Module, model))


def _unwrap_torch_compile_submodules(module: nn.Module) -> nn.Module:
    if isinstance(module, torch._dynamo.OptimizedModule):
        return cast(nn.Module, module._orig_mod)

    for name, child in module.named_children():
        unwrapped_child = _unwrap_torch_compile_submodules(child)
        if unwrapped_child is not child:
            module._modules[name] = unwrapped_child

    return module
