class DFA:

    def __init__(self, dfa: [str]) -> None:
        '''Creates this DFA as described in dfa.

        Parameters:
        - self: obligatory reference to this object
        - dfa: description of the desired DFA

        Return:
        None
        '''
        self.Q = dfa[0].split(' ')
        self.Sigma = dfa[1].split(' ')
        self.q0 = dfa[2]
        self.F = dfa[3].split(' ')
        self.delta = {}
        for i in range(4, len(dfa)):
            one_map = dfa[i].split(' ')
            self.delta[(one_map[0], one_map[1])] = one_map[2]

    def accepts(self, w: str) -> bool:
        '''Returns the acceptance status of w.

        Parameters:
        - self: obligatory reference to this object
        - w: the string to check

        Return:
        True if this DFA accepts w, False otherwise.
        '''
        current_state = self.q0
        for w_ in w:
            current_state = self.delta[(current_state, w_)]
        return current_state in self.F
