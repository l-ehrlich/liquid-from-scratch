let
  pkgs = import <nixpkgs> {
      config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
  python = pkgs.python312;
  pythonPackages = python.pkgs;
  lib-path = with pkgs; lib.makeLibraryPath [
    stdenv.cc.cc
  ];
in with pkgs; mkShell {
  packages = [
    python
    pythonPackages.pip
    pythonPackages.virtualenv
    cudaPackages.cudnn
    cudaPackages.cudatoolkit
  ];

  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}:/run/opengl-driver/lib
    export CUDA_PATH=${cudaPackages.cudatoolkit}/lib
    VENV=.venv

    if test ! -d $VENV; then
      python3.12 -m venv $VENV
    fi
    source ./$VENV/bin/activate
    export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
    pip install -r requirements.txt
  '';

  postShellHook = ''
    ln -sf ${python.sitePackages}/* ./.venv/lib/python3.12/site-packages
  '';
}