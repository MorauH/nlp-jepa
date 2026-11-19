{
  description = "JEPA language model (Python/uv, PyTorch/HuggingFace)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.python312
            # Inductor's C++ codegen shells out to `openssl` when it hashes
            # precompiled headers for torch.compile.
            pkgs.openssl.bin
          ];

          LD_LIBRARY_PATH = lib.makeLibraryPath ([
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.gcc-unwrapped.lib
          ] ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            pkgs.libGL
            pkgs.libx11
          ]);

          shellHook = ''
            if [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi
            # Triton discovers libcuda through /sbin/ldconfig, which NixOS does
            # not have. Point it at the WSL/NixOS driver directory instead.
            for _cuda_dir in /usr/lib/wsl/lib /run/opengl-driver/lib; do
              if [ -e "$_cuda_dir/libcuda.so.1" ]; then
                export TRITON_LIBCUDA_PATH="$_cuda_dir"
                break
              fi
            done
            unset _cuda_dir
            if [ ! -d .venv ] && [ -f pyproject.toml ]; then
              echo "Creating uv virtual environment..."
              uv sync
            fi
            if [ -f .venv/bin/activate ]; then
              source .venv/bin/activate
            fi
            echo "JEPA environment ready. Run: python -m jepa train"
          '';
        };
      });
}
