addi x22, x0, 0 # i == 0
addi x24, x0, 10 # i == 10
addi t2, x0, 0 # array offset

loop1:
beq x22, x24, exitLoop1
sw x22, 0x200(t2)
addi x22, x22, 1
addi t2, t2, 4
beq x0, x0, loop1
exitLoop1:

addi x23, x0, 0 # sum == 0
addi x22, x0, 0 # i == 0
addi t2, x0, 0 # array offset

loop2: 
beq x22, x24, exitLoop2
lw t0, 0x200(t2)
add x23, x23, t0
addi x22, x22, 1
addi t2, t2, 4
beq x0, x0, loop2

exitLoop2: