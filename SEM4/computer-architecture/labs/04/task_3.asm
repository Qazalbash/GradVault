li x10, 5
jal x1, ntri

j exit

ntri:
addi x2, x2, -8  # adjust stack for 2 items
sw x1, 4(x2)     # save the return address
sw x10, 0(x2)    # save the argument n
 
addi x5, x10, -1 # x5 = n - 1
bgt x5, x0, L1   # if (n - 1) > 0, go to L1

addi x10, x0, 1  # return 1
addi x2, x2, 8   # pop 2 items off stack
jalr x0, 0(x1)   # return to caller


L1: addi x10, x10, -1 # n >= 1: argument gets (n − 1)
jal x1, ntri          # call ntri with (n − 1)
 
addi x6, x10, 0 # return from jal: move result of ntri (n - 1) to x6:
lw x10, 0(x2)   # restore argument n
lw x1, 4(x2)    # restore the return address
addi x2, x2, 8  # adjust stack pointer to pop 2 items

add x10, x10, x6 # return n + ntri (n − 1)

jalr x0, 0(x1) # return to the caller

exit:
mv x11, x10
addi x10, x0, 1
ecall