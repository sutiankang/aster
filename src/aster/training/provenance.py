"""Bounded declarations for the last successful training phase."""

from copy import deepcopy
import math


def _finite_json(value):
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _finite_json(item) for key, item in value.items())
    return False


def validate_update_record(record, *, role, updates):

    if not isinstance(role, str) or not role or type(updates) is not int or updates < 0:
        raise ValueError("Successful update provenance requires a valid role/update clock")
    if record is None:
        return None
    if (
        not isinstance(record, dict)
        or set(record) != {"role", "role_updates", "phase", "objective_configuration"}
        or record["role"] != role
        or type(record["role_updates"]) is not int
        or record["role_updates"] < 1
        or record["role_updates"] != updates
        or not isinstance(record["phase"], str)
        or not record["phase"]
    ):
        raise ValueError("Successful update provenance has invalid role/update clock/phase")
    descriptor = record["objective_configuration"]
    if descriptor is not None:
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"class", "codec", "configuration"}
            or not isinstance(descriptor["class"], str)
            or not descriptor["class"]
            or descriptor["codec"] not in {"config_dict", "to_dict"}
            or not isinstance(descriptor["configuration"], dict)
            or not _finite_json(descriptor["configuration"])
        ):
            raise ValueError(
                "Successful update objective must be an explicit finite JSON descriptor"
            )
    return deepcopy(record)
