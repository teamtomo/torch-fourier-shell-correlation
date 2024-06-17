"""Compute the Fourier shell correlation in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-fourier-shell-correlation")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .fsc import fsc
