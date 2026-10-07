import numpy as np
import pytest
import torch

from qenas.models.chromosome import InvalidChromosomeError, SEARCH_SPACE_SIZE, decode, validate_chromosome
from qenas.models.darts_cell import GENOTYPES, Genotype, validate_genotype
from qenas.models.network import QENASNet, count_parameters


def test_search_space_size():
    assert SEARCH_SPACE_SIZE == 5 ** 8 == 390625


@pytest.mark.parametrize("bad", [[0] * 7, [0] * 9, [0, 1, 2, 3, 4, 5, 0, 0], [-1] + [0] * 7, [0.5] + [0] * 7])
def test_invalid_chromosomes_rejected(bad):
    with pytest.raises(InvalidChromosomeError):
        validate_chromosome(bad)


def test_decode_order_matches_mixed_ggnas_numbering():
    assert decode([0, 1, 2, 3, 4, 0, 1, 2])[:5] == ["Residual", "Dense", "Inception", "ConvNeXt", "DARTS"]


@pytest.mark.parametrize("genotype", sorted(GENOTYPES))
@pytest.mark.parametrize("hw", [(64, 64), (64, 96)])
def test_every_block_at_every_position(genotype, hw):
    torch.manual_seed(0)
    for g in range(5):
        m = QENASNet([g] * 8, base_channels=16, genotype_name=genotype)
        y = m(torch.randn(2, 3, *hw))
        assert y.shape == (2, 1, *hw)
        assert torch.isfinite(y).all()


def test_random_mixed_chromosomes_and_scales():
    rng = np.random.default_rng(0)
    for scale in (3, 5, 7):
        for _ in range(4):
            ch = rng.integers(0, 5, 8).tolist()
            m = QENASNet(ch, base_channels=16, scale=scale)
            assert m(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)
            assert count_parameters(m) > 0


def test_invalid_genotype_rejected():
    bad = Genotype(down=[("dil_conv", 0), ("down_conv", 1)], down_concat=[2],
                   up=[("up_conv", 1), ("dil_conv", 0)], up_concat=[2])
    with pytest.raises(ValueError):
        validate_genotype(bad)   # a resolution-preserving op on a down-cell input breaks spatial consistency
