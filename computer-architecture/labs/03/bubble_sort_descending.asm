addi x22, x0, 9 # i = 9
addi x24, x0, -1 # -1
addi x7, x0, 36 # array offset

loop1:

beq x22, x24, exitloop1
sub x2, x22, x7
sw x2, 0x200(x7)
addi x22, x22, -1
addi x7, x7, -4
beq x0, x0, loop1

exitloop1:

addi x8, x0, 0 # array offset for loop10
addi x9, x0, 0 # array offset for loop2
addi x22, x0, 0
addi x24, x0, 10 # 10

loop10:
beq x22, x24, exitloop10
add x23, x0, x0 # j = 0
addi x9, x0, 0 


loop2:
beq x23, x24, exitloop2

lw x3, 0x200(x8) # a[i]
lw x4, 0x200(x9) # a[j]

bge x3, x4, exitIf

lw x5, 0x200(x8) # int temp = a[i]
sw x4, 0x200(x8) # a[i] = a[j]
sw x5, 0x200(x9) # a[j] = temp

exitIf:
addi x23, x23, 1
addi x9, x9, 4
beq x0, x0, loop2

exitloop2:

addi x22, x22, 1
addi x8, x8, 4
beq x0, x0, loop10

exitloop10: