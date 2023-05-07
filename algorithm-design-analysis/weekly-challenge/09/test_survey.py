import pytest
import hashlib


def survey_campus(fname: str) -> (int, int, int):
    ''' Returns information of an optimal tour of the network stored at fname.

    Parameters:
    - fname: the path to information on the network

    Returns:
    information of an optimal tour of the network.
    '''
    pass


HASHES = [
    '35990e4e99eb8e2eac6c9b4f1eccaf5a981ae646699d39de735dd74020f4d61e',
    '3961b95f632b05c173ddd6535ea1033a2fc153e88bd576ec6179c5f5941cec83',
    '4f867de29115ef2def65827c919291b01ffa0ec2b02464639dd79af0953c40cd',
    'ec3c07852b2e316d8611f821f3017b9e99c947b3504fcb605dc3a0a0510d1d42',
    '21a1fefbbcaabbd404826d81e87cc823400d06a442b7c216852c9cdbaea036f0',
    'e1bef7ddc49512f8488a0af2c7cffa49ff7d8f1c83ccf7aad1042cde2c0a5293',
    '1218c28d637dafabcff5ba43cf7d5eb514097ca05223616a178b271654f8a287'
]


def hashcode(n: int) -> str:
    return hashlib.sha256(str(n).encode('utf-8')).hexdigest()


@pytest.mark.parametrize("i", range(len(HASHES)))
def test_survey(i: int):
    fname = f'tests/survey_{i}.txt'
    assert hashcode(survey_campus(fname)) == HASHES[i], \
        f'Test failed for {fname}'
