import pytest
import numpy as np
import matplotlib.pyplot as plt
from cp_measure.examples import get_pixels, get_masks
from cp_measure.core.measuregranularity import get_granularity


@pytest.fixture
def pixels() -> np.ndarray:
    return get_pixels()


@pytest.fixture
def mask() -> np.ndarray:
    return get_masks()["center"]


@pytest.fixture
def real_pixels() -> np.ndarray:
    return plt.imread(
        "../data/source_13__20220914_Run1__CP-CC9-R1-01__I13__5/SLFN13_01_AGP__source_13__20220914_Run1__CP-CC9-R1-01__I13__5.tif")


@pytest.fixture
def real_mask() -> np.ndarray:
    return plt.imread("../data/source_13__20220914_Run1__CP-CC9-R1-01__I13__5/cytosol_mask.tif")


def test_basic_granularity(pixels, mask):
    granularity = get_granularity(mask=mask, pixels=pixels)
    assert granularity is not None


def test_real_granularity(real_pixels, real_mask):
    granularity = get_granularity(mask=real_mask, pixels=real_pixels)
    assert granularity is not None

