"""
    Automatically include all python files in this directory in __all__
    This means that simply adding a python file will mean it can be
    imported using 'from op_kernels import *'
"""
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
