import pytest

from volrpynn.model import Model

def test_empty_model():
    with pytest.raises(Exception):
        Model([], [], [], None)
