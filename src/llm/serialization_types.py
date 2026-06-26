from __future__ import annotations

from typing import Any, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

ConfigPayload: TypeAlias = dict[str, Any]
CheckpointPayload: TypeAlias = dict[str, Any]
CheckpointState: TypeAlias = dict[str, Any]
ConfigDrift: TypeAlias = dict[str, dict[str, Any]]
