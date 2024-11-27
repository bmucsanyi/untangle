"""Default context manager that does nothing."""

from types import TracebackType


class DefaultContext:
    """Identity context manager that does nothing."""

    def __enter__(self) -> None:
        """Enters the context manager."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exits the context manager."""
        del exc_type, exc_value, traceback

    def __call__(self) -> "DefaultContext":
        """Call method that returns self."""
        return self
