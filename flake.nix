{
  description = "Python + CUDA Nix flake template for ML/dev environments (manual venv with Poetry)";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # ─── Centralized version configuration ───
        pythonVersion = "python311";       # e.g. "python312"
        cudaVersion   = "cudaPackages_12_6"; # e.g. "cudaPackages_11_8"
        venvDir       = ".venv";

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        python = pkgs.${pythonVersion + "Full"};
        cuda   = pkgs.${cudaVersion};

        pythonPackages = pkgs.${pythonVersion}.pkgs;

        inherit (pkgs.linuxPackages) nvidia_x11;

      in {
        devShells.default = pkgs.mkShell {
          name = "python-cuda-dev";

          buildInputs = [
            python
            pythonPackages.venvShellHook
          ];

          packages = [
            pkgs.poetry
            pkgs.jupyter
          ];

          inherit venvDir;

          # ensure poetry installs to the correct location (NOT NIX STORE)
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH

            if [ ! -f pyproject.toml ]; then
              echo "No pyproject.toml found. Running 'poetry init' to create one."
              poetry init
            fi

            poetry env use .venv/bin/python || true
            poetry install --no-root || true

          '';

          # ensure linker can find required C/C++ libraries and CUDA runtime while always preferring the host driver
          postShellHook = ''
            unset SOURCE_DATE_EPOCH
            export CUDA_PATH=${cuda.cudatoolkit.lib}
            # Host driver first, then toolkit libs, then any existing value (open gl points cuda at the global installation, since the shell doesn't fully create its own)
	    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${cuda.cudatoolkit.lib}/lib:${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}:$LD_LIBRARY_PATH"

	    # may not be needed
            export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-i/usr/include"          

            poetry env info || true
          '';

        };
      });
}

