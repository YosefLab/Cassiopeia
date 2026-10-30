import contextlib
import logging
import sys


class _NamespaceFormatter(logging.Formatter):
    def __init__(self, owner: "CassiopeiaLogger"):
        super().__init__("[%(asctime)s] %(levelname)7s [%(namespace)s] %(message)s")
        self._owner = owner

    def format(self, record):
        record.namespace = self._owner.namespace
        return super().format(record)


class CassiopeiaLogger(logging.Logger):
    """Logger whose messages are prefixed with the current namespace.

    ``logger.namespaced(name)`` works as a decorator or context manager and
    sets the namespace shown in log messages.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.namespace = "main"
        # console handler (``ch``, as in the ngs-tools logger this replaces)
        self.ch = logging.StreamHandler(sys.stderr)
        self.ch.setFormatter(_NamespaceFormatter(self))
        self.addHandler(self.ch)
        self.propagate = False

    def namespaced(self, namespace: str):
        """Decorator / context manager that sets the log namespace."""
        logger = self

        class _Namespace(contextlib.ContextDecorator):
            def __enter__(self):
                self.previous = logger.namespace
                logger.namespace = namespace
                return self

            def __exit__(self, *exc):
                logger.namespace = self.previous
                return False

        return _Namespace()


logger = CassiopeiaLogger("cassiopeia")
logger.setLevel(logging.INFO)
