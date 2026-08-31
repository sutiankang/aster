"""Fixed Python entry point for native and torchrun workers."""

from aster.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
