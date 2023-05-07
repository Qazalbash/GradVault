import pytest
import random
import string
import sys

sys.path.append(".")

from hello import greet


@pytest.mark.parametrize('_', range(100))
def test_greet(_):
    name = ''.join(random.choices(string.printable, k=random.randint(1, 1000)))
    greeting = greet(name)
    assert greeting == f'Hello, {name}!', \
        f'Bad {greeting=} for {name=}'
