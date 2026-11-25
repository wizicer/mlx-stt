from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# MLX locates mlx.metallib relative to the loaded dylib. Preserve the
# package layout (mlx/lib) instead of flattening the files into the bundle root.
binaries = collect_dynamic_libs("mlx")
datas = collect_data_files("mlx", includes=["lib/*.metallib"])
hiddenimports = ["mlx._reprlib_fix", "mlx.__array_api_info", "mlx.nn", "mlx.utils"]
