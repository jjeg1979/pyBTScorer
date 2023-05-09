"""Tests that btgenbox raises exceptions appropiately"""

# Standard library imports
from pathlib import Path

# Non-standard library imports
import pytest

# Project imports

from src.parser.btparser import (
    BtOrderType,    
)

from src.parser.btgenbox import BtGenbox


def test_btgenbox_init_raises_fileNotFoundError():
    with pytest.raises(FileNotFoundError) as er:
        BtGenbox(Path(r'src/payload'), 'fileNotExistent')
    assert er.type == FileNotFoundError

   
