from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

ConfigPayload: TypeAlias = dict[str, Any]
CheckpointPayload: TypeAlias = dict[str, Any]
CheckpointState: TypeAlias = dict[str, Any]
ConfigDrift: TypeAlias = dict[str, dict[str, Any]]


@runtime_checkable
class Serializable(Protocol):
    """Anything that can render itself to a JSON-ready mapping.

    Implemented by ``ModelConfig``, ``TrainConfig``, ``RunConfig``,
    ``EvalResult``, and ``Checkpoint``. This protocol intentionally covers only
    the instance-side ``toDict`` contract, which is uniform across all of them.

    The inverse (``fromDict``) is deliberately *not* part of the protocol: it is
    a constructor-like operation called on the class rather than an instance,
    and its form is not uniform (some classes use ``@classmethod``, ``Checkpoint``
    uses ``@staticmethod``). Keeping ``fromDict`` out keeps the protocol honest
    and lets each class deserialize in whatever way is clearest for it.

    ``runtime_checkable`` allows ``isinstance(x, Serializable)`` for the
    ``toDict`` method, though static typing is the intended use.
    """

    def toDict(self) -> dict[str, Any]: ...
