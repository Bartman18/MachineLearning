import pandas as pd
import pytest



@pytest.fixture
def data():
    return 'excel1.xlsx'

def test_read(data):
    df = pd.read_excel(data)


    assert not df.empty

