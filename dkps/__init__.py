"""dkps -- perspective spaces for comparing black-box models from their cached responses.

Estimators (all share fit(records) / predict(records) / update(records); see dkps.records
for the record format):

    PKPS, DKPS          response-based: product-kernel perspective space -> MDS -> k-NN
    SampleScore         the mean of a cell's observed per-response scores
    IRT                 1PL item-response baseline (binary tasks)
    LRMC                low-rank matrix completion of the sample-score matrix
    Ensemble            two-member blend with a learned weight

Pre-processing: Whitener (logit / standardize / two-way bias), FrozenPCA,
BlockDiagonalEmbedding. Lower-level pieces used by the experiment scripts remain
available: ProductKernelPerspectiveSpace (distance matrices over a bandwidth grid),
DataKernelPerspectiveSpace (the original paired DKPS), matrix_completion_predict and
the irt_* functions, generate_benchmark_data (synthetic benchmark).
"""

from .pkps import PKPS, DKPS
from .dkps import DataKernelPerspectiveSpace
from .synthetic import generate_benchmark_data
from .unpaired_dkps import ProductKernelPerspectiveSpace, DoubleKernelDKPS
from . import eee
from .ensemble import Ensemble
from .preprocessing import Whitener, FrozenPCA, BlockDiagonalEmbedding
from .baselines import (
    SampleScore,
    IRT,
    LRMC,
    matrix_completion_predict,
    irt_fit_difficulties,
    irt_estimate_ability,
    irt_predict,
)

__all__ = [
    'PKPS',
    'DKPS',
    'DataKernelPerspectiveSpace',
    'ProductKernelPerspectiveSpace',
    'DoubleKernelDKPS',
    'eee',
    'Ensemble',
    'SampleScore',
    'IRT',
    'LRMC',
    'Whitener',
    'FrozenPCA',
    'BlockDiagonalEmbedding',
    'generate_benchmark_data',
    'matrix_completion_predict',
    'irt_fit_difficulties',
    'irt_estimate_ability',
    'irt_predict',
]
