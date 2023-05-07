def greet(name: str) -> str:
    '''Returns a greeting for name.

    The greeting is "Hello, " followed by the name and an exclamation mark. For
    example, the greeting for "Wasif" is "Hello, Wasif!".

    Parameters:
    - name: the name to be greeted.

    Constraints:
    - 1 <= len(name) <= 1000
    - all(isprintable(i) for i in name)

    Returns:
    the greeting for name.
    '''
    return f"Hello, {name}!"
