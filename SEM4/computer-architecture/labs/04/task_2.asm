li x12, 10 # a[0] = 10
lui x13, 0 # array offset

# initializing array

sw x12, 0x100(x13)
addi x12, x12, -1
addi x13, x13, 4
sw x12, 0x100(x13)

addi x12, x12, -1
addi x13, x13, 4
sw x12, 0x100(x13)

addi x12, x12, 1
addi x13, x13, 4
sw x12, 0x100(x13)

addi x12, x12, -4
addi x13, x13, 4
sw x12, 0x100(x13)

addi x12, x12, -15
addi x13, x13, 4
sw x12, 0x100(x13)

addi x12, x12, 12
addi x13, x13, 4
sw x12, 0x100(x13)

addi x12, x12, -5
addi x13, x13, 4
sw x12, 0x100(x13)

# end initializing

li x10, 0x100
addi x11, x0, 8

jal bubble
j exit

bubble:

beq x10, x0, exitloop10
beq x11, x0, exitloop10

addi x8, x0, 0 # array offset for loop10
addi x9, x0, 0 # array offset for loop2
addi x22, x0, 0

loop10:
beq x22, x11, exitloop10
add x23, x22, x0 # j = i
addi x9, x8, 0 

loop2:
beq x23, x11, exitloop2

add x6, x8, x10
lw x3, 0(x6) # a[i]
add x7, x9, x10
lw x4, 0(x7) # a[j]

bge x3, x4, exitIf

lw x5, 0(x6) # int temp = a[i]
sw x4, 0(x6) # a[i] = a[j]
sw x5, 0(x7) # a[j] = temp

exitIf:
addi x23, x23, 1
addi x9, x9, 4
beq x0, x0, loop2

exitloop2:

addi x22, x22, 1
addi x8, x8, 4
beq x0, x0, loop10

exitloop10:

exit:
