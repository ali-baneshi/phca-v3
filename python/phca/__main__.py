"""Allow `python -m phca` as well as the `phca` console script."""

from phca.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
