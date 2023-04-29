# ----------------------------------------------------------------------------------------------

# A list of some of the selection method combinations
selection_schemes_cases = [
    (0, 4),  # ▪ FPS and Random
    (2, 3),  # ▪ Binary Tournament and Truncation
    (3, 3),  # ▪ Truncation and Truncation
    (4, 4),  # ▪ Random and Random
    (0, 3),  # ▪ FPS and Truncation
    (1, 2),  # ▪ RBS and Binary Tournament
    (4, 3),  # ▪ Random and Truncation
    (2, 0),  # ▪ Binary Tournament and FPS
    (2, 1),  # ▪ Binary Tournament and RBS
    (3, 0),  # ▪ Truncation and FPS
]

# ----------------------------------------------------------------------------------------------
"""
- Selection Methods
    0. Fitness Proportional Selection
    1. Rank based Selection
    2. Binary Tournament
    3. Truncation
    4. Random

- selection cases to test
    selection_case 0: ▪ FPS and Random  
    selection_case 1: ▪ Binary Tournament and Truncation 
    selection_case 2: ▪ Truncation and Truncation
    selection_case 3: ▪ Random and Random
    selection_case 4: ▪ FPS and Truncation
    selection_case 5: ▪ RBS and Binary Tournament
    selection_case 6: ▪ Random and Truncation
    selection_case 7: ▪ Binary Tournament and FPS 
    selection_case 8: ▪ Binary Tournament and RBS
    selection_case 9: ▪ Truncation and FPS
"""

selection_case0 = selection_schemes_cases[0]
selection_case1 = selection_schemes_cases[1]
selection_case2 = selection_schemes_cases[2]
selection_case3 = selection_schemes_cases[3]
selection_case4 = selection_schemes_cases[4]
selection_case5 = selection_schemes_cases[5]
selection_case6 = selection_schemes_cases[6]
selection_case7 = selection_schemes_cases[7]
selection_case8 = selection_schemes_cases[8]
selection_case9 = selection_schemes_cases[9]

selection_cases = [
    selection_case0, selection_case1, selection_case2, selection_case3,
    selection_case4, selection_case5, selection_case6, selection_case7,
    selection_case8, selection_case9
]
