"""Pinned catalogue ownership must support safe retained native fingerprints."""

from dataclasses import fields

import numpy as np
import pytest

from spacepdhcg.gtoc12.data import AsteroidCatalogue


def test_immutable_catalogue_owns_values_and_cannot_be_made_writable():
    original = AsteroidCatalogue(
        ids=np.arange(1, 4, dtype=np.int64),
        **{
            field.name: np.arange(3, dtype=np.float64)
            for field in fields(AsteroidCatalogue)
            if field.name not in ("ids", "source_sha256")
        },
        source_sha256="source identity",
    )
    frozen = original.immutable_copy()
    assert frozen.source_sha256 == original.source_sha256
    for field in fields(original):
        if field.name == "source_sha256":
            continue
        source, target = getattr(original, field.name), getattr(frozen, field.name)
        assert target.dtype == source.dtype and target.shape == source.shape
        np.testing.assert_array_equal(target, source)
        assert type(target.base) is bytes and not target.flags.writeable
        with pytest.raises(ValueError):
            target.setflags(write=True)
        with pytest.raises(ValueError):
            target[0] = 99
        source[0] += 1
        assert target[0] != source[0]
