"""The pre-PKPS dkps API (DataKernelPerspectiveSpace taking a dict of arrays,
DataKernelFunctionalSpace, UnpairedDKPS, knn_graph utilities), kept so the
agentic-traces scripts in scripts/legacy keep working after the fork merge."""
from .dkps import DataKernelPerspectiveSpace, DataKernelFunctionalSpace
from .unpaired_dkps import UnpairedDKPS
