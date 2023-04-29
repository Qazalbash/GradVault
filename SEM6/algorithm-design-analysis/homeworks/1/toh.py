def Tower_of_Habib(state):
    mid = len(state)//2
    if len(state) > 1:
        out = Tower_of_Habib(state[:mid])
    else:
        if state[0] == 1:
            return True
        else:
            return False
    return out
        