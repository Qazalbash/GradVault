li x10, 3
li x11, 5
jal x1, sum

mv x11, x10
addi x10, x0, 1
ecall

j exit

sum:
add x10, x10, x11
jalr x0, 0(x1)

exit: