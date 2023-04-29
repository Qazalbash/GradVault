`include "immem.v"
`include "adder.v"
`include "pc.v"

module Instruction_Fetch(
	input clk,
  	input reset,
  output [31:0] Instruction 
);
  
  wire [63:0] PC_in;
  wire [63:0] PC_out;
  
  
   Instruction_Memory I(
       .Inst_Address(PC_out),
       .Instruction(Instruction)
	);
  
  Adder A(
       .a(PC_out),
      .b(64'b100),
      .c(PC_in)
	);
  
  Program_Counter PC(
      .clk(clk),
      .reset(reset),
      .PC_in(PC_in),
      .PC_out(PC_out)
	);
  
endmodule