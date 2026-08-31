"""Strict logical-to-public parameter-name mapping."""


def _export(module, state, prefix, metadata):
    for internal, public in module._aster_parameter_key_map.items():
        if prefix + internal in state:
            if prefix + public in state:
                raise ValueError("Public parameter mapping collision")
            state[prefix + public] = state.pop(prefix + internal)


def _import(module, state, prefix, metadata, strict, missing, unexpected, errors):
    for internal, public in module._aster_parameter_key_map.items():
        if prefix + public in state:
            if prefix + internal in state:
                raise ValueError("Checkpoint has both internal and public parameter keys")
            state[prefix + internal] = state.pop(prefix + public)


def register_parameter_codec(module, mapping):
    mapping = dict(mapping)
    if (
        not mapping
        or len(set(mapping.values())) != len(mapping)
        or any(not a or not b or a == b for a, b in mapping.items())
    ):
        raise ValueError("Parameter codec must be an explicit nontrivial bijection")
    if hasattr(module, "_aster_parameter_key_map"):
        raise ValueError("Parameter codec already registered")
    module._aster_parameter_key_map = mapping
    module.register_state_dict_post_hook(_export)
    module.register_load_state_dict_pre_hook(_import)


def public_parameter_names(module):

    result = {name: name for name, _ in module.named_parameters()}

    for prefix, child in reversed(list(module.named_modules())):
        for internal, public in getattr(child, "_aster_parameter_key_map", {}).items():
            old = f"{prefix}.{internal}" if prefix else internal
            new = f"{prefix}.{public}" if prefix else public
            for name, current in result.items():
                if current == old:
                    result[name] = new
    if len(set(result.values())) != len(result):
        raise ValueError("Parameter codec produced ambiguous public names")
    return result
