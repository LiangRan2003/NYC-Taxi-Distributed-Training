import sys
import types


class _FakeMPI:
    SUM = object()


mpi4py_module = types.ModuleType("mpi4py")
mpi4py_module.MPI = _FakeMPI
sys.modules.setdefault("mpi4py", mpi4py_module)
