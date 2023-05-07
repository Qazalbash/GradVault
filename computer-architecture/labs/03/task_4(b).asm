addi x22, x0, 1 # b
addi x23, x0, 2 # c
addi t3, x0, 2 # nmber 2
addi x20, x0, 3 # x == 3

addi t2, x0, 1 # x == 1
beq t2, x20, case1
addi t2, t2, 1      # x == 2
beq t2, x20, case2
addi t2, t2, 1      # x == 3
beq t2, x20, case3
addi t2, t2, 1      # x == 4
beq t2, x20, case4
addi x21, x0, 0     # default case
beq x0, x0, exit

case1: add x21, x22, x23      # a=b+c
beq x0, x0, exit
case2: sub x21, x22, x23      # a=b-c
beq x0, x0, exit
case3: mul x21, x22, t3       # a=b*2
beq x0, x0, exit
case4: div x21, x22, t3       # a=b/2
beq x0, x0, exit

exit: