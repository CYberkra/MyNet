"""Tests for model_interfaces.py — PGDAOutput, make_pgda_output, unpack_pgda_output."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))

import torch
import pytest
from pgdacsnet.model_interfaces import PGDAOutput, make_pgda_output, unpack_pgda_output


@pytest.fixture
def t():
    return torch.randn(2, 1, 128, 64), torch.randn(2, 64), torch.randn(2, 1, 128, 64)


class TestPGDAOutput:
    def test_construct(self, t):
        o = PGDAOutput(*t)
        assert o.mask_logits is t[0]
        assert o.presence_logits is t[1]
        assert o.center_logits is t[2]
        assert len(o) == 3

    def test_tuple_unpack(self, t):
        m, p, c = PGDAOutput(*t)
        assert m is t[0] and p is t[1] and c is t[2]

    def test_dict_access(self, t):
        o = PGDAOutput(*t)
        assert o['mask_logits'] is t[0]
        assert o['presence_logits'] is t[1]
        assert o['center_logits'] is t[2]

    def test_attribute_access(self, t):
        o = PGDAOutput(*t)
        assert o.mask_logits is t[0]

    def test_contains(self, t):
        o = PGDAOutput(*t)
        assert 'mask_logits' in o and 'nonexistent' not in o


class TestMakePGDAOutput:
    def test_make_three(self, t):
        o = make_pgda_output(*t)
        assert isinstance(o, PGDAOutput)

    def test_make_two(self, t):
        o = make_pgda_output(t[0], t[1])
        assert o.center_logits is None


class TestUnpackPGDAOutput:
    def test_from_pgda_output(self, t):
        m, p, c = unpack_pgda_output(PGDAOutput(*t))
        assert m is t[0] and p is t[1] and c is t[2]

    def test_from_dict(self, t):
        m, p, c = unpack_pgda_output({'mask_logits': t[0], 'presence_logits': t[1], 'center_logits': t[2]})
        assert m is t[0] and p is t[1] and c is t[2]

    def test_from_3tuple(self, t):
        m, p, c = unpack_pgda_output(t)
        assert m is t[0] and p is t[1] and c is t[2]

    def test_from_2tuple(self, t):
        m, p, c = unpack_pgda_output((t[0], t[1]))
        assert m is t[0] and p is t[1] and c is None

    def test_invalid_raises(self):
        with pytest.raises(TypeError):
            unpack_pgda_output(42)
