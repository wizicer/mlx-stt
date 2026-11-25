from multiprocessing import freeze_support

from mlx_stt.cli import main


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
