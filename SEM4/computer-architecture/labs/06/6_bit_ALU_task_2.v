`include "ALU_1_bit.v"

module ALU_6_bit
  (
    input [5:0] a,
    input [5:0] b,
    input [3:0] ALUop,
    input cin,
    
    output cout,
    output [5:0] result
  );
  
  wire cout1;
  wire cout2;
  wire cout3;
  wire cout4;
  wire cout5;
  
  ALU_1_bit I0(
    .a(a[0]),
    .b(b[0]),
    .ALUop(ALUop),
    .cin(cin),
    .cout(cout1),
    .result(result[0])
  );
  
   ALU_1_bit I1(
     .a(a[1]),
     .b(b[1]),
    .ALUop(ALUop),
     .cin(cout1),
     .cout(cout2),
     .result(result[1])
  );
  
   ALU_1_bit I2(
     .a(a[2]),
     .b(b[2]),
    .ALUop(ALUop),
     .cin(cout2),
     .cout(cout3),
     .result(result[2])
  );
  
   ALU_1_bit I3(
     .a(a[3]),
     .b(b[3]),
    .ALUop(ALUop),
     .cin(cout3),
     .cout(cout4),
     .result(result[3])
  );
  
   ALU_1_bit I4(
     .a(a[4]),
     .b(b[4]),
    .ALUop(ALUop),
     .cin(cout4),
     .cout(cout5),
     .result(result[4])
  );
  
   ALU_1_bit I5(
     .a(a[5]),
     .b(b[5]),
    .ALUop(ALUop),
     .cin(cout5),
     .cout(cout),
     .result(result[5])
  );
  
  
endmodule
  