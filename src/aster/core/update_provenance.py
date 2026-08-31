"""Validate the declared objective and roles of the last successful training update."""

import copy

from .serialization import digest_json


def validate_successful_update_record(record, descriptor, *, role="model", role_updates):

    required = {"role", "role_updates", "phase", "objective_configuration"}
    if (
        not isinstance(record, dict)
        or set(record) != required
        or not isinstance(descriptor, dict)
        or set(descriptor) != {"class", "codec", "configuration"}
    ):
        raise ValueError("Actual successful objective provenance is unavailable")
    if (
        not isinstance(descriptor["class"], str)
        or not descriptor["class"]
        or descriptor["codec"] not in {"config_dict", "to_dict"}
        or not isinstance(descriptor["configuration"], dict)
    ):
        raise ValueError("Actual successful objective descriptor is malformed")
    if (
        record["role"] != role
        or type(role_updates) is not int
        or type(record["role_updates"]) is not int
        or record["role_updates"] < 1
        or record["role_updates"] != role_updates
        or not isinstance(record["phase"], str)
        or not record["phase"]
    ):
        raise ValueError("Actual successful objective provenance has a stale role/update clock")
    if record["objective_configuration"] is None or digest_json(
        record["objective_configuration"]
    ) != digest_json(descriptor):
        raise ValueError(
            "Actual successful phase objective differs from the declared publishing objective"
        )
    return copy.deepcopy(record)
