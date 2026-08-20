"""
Non-functional test — Chapter 10.2 row 3 (Supportability / Extension Test).

Adds a new generator, DummyGen (generators/dummy_gen.py), inheriting from
BaseGenerator with a minimal working run(), and registers it in the real
generator pipeline (generators/unified_generator.py — UnifiedGenerator is the
"dispatcher: routes gen_type to the right generator", per CLAUDE.md's
repository map) without touching UnifiedGenerator.generate_stego() or any
other dispatch/manager logic.

Runs fast, needs no external data — part of the default `pytest` run.
"""

import numpy as np
import pytest

from generators.base_generator import BaseGenerator
from generators.dummy_gen import DummyGen
from generators.unified_generator import UnifiedGenerator


def test_dummygen_is_a_base_generator():
    assert issubclass(DummyGen, BaseGenerator)


def test_unified_generator_registers_dummy_without_dispatch_changes():
    gen = UnifiedGenerator()
    assert "dummy" in gen.generators
    assert isinstance(gen.generators["dummy"], DummyGen)


def test_unified_generator_can_invoke_dummy(cover_array):
    gen = UnifiedGenerator()
    stego, metric = gen.generate_stego(cover_array, None, {"gen_type": "dummy"})
    assert stego is not None
    assert stego.shape == cover_array.shape
    assert stego.dtype == np.uint8
    # DummyGen flips the LSB of every pixel — verify it actually ran, not a passthrough.
    assert np.array_equal(stego, cover_array ^ 1)


def test_unified_generator_unknown_type_still_handled_generically():
    """generate_stego()'s fallback path is untouched by the dummy registration."""
    gen = UnifiedGenerator()
    stego, metric = gen.generate_stego(np.zeros((8, 8), dtype=np.uint8), None,
                                        {"gen_type": "not_a_real_method"})
    assert stego is None
    assert metric == 0
