"""Default context manager that does nothing."""


class DefaultContext:
    """Identity context manager."""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self):
        return self
