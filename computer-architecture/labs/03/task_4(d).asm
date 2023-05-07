addi x5, x0, 2 # a
addi x6, x0, 2 # b
addi x7, x0, 0 # i == 0
addi x10, x0, 0x100

loop1:
beq x7, x5, exitLoop1 # i == a
addi x8, x0, 0 # array offset
addi x29, x0, 0 # j == 0

loop2: 
beq x29, x6, exitLoop2
add x9, x7, x29 # i + j
add x11, x10, x8 
sw x9, 0(x11)
addi x29, x29, 1 # j++
addi x8, x8, 16
beq x0, x0, loop2
exitLoop2:

addi x7, x7, 1 # i++
beq x0, x0, loop1
exitLoop1: