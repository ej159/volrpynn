import pytest

from volrpynn.model import *

def test_empty_model():
    with pytest.raises(Exception):
        Model([], [], [], None)
