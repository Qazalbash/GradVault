module MUX
  (
    input [63:0] A,
    input [63:0] B,
    input S,
    output [63:0] Y
  );
  
  //The 4 raw data inputs are treated as 7 bit vectors, with a 2 bit vector input for control/selector bits; the output is also a 7 bit vector.

  assign Y = S ? B : A;
  
  //In the same way that the conditional operator is used in C, we utilise it in a hierarchical fashion. Whether S[1] is on(1), we check if S[0] is on(1), and if it is, the output Y is allocated D, otherwise it is assigned C. If S[1] is not on(1), we check if S[0] is on(1), and if it is, we assign B to the output Y, otherwise we assign A.
  
endmodule