`include "ALU_control_unit.v"
`include "control_unit.v"

module Top_Control(
    input [6:0] Opcode,
    input [3:0] Funct,
    output Branch, 
    output MemRead,
    output MemtoReg,
    output MemWrite,
    output ALUSrc, 
    output RegWrite,
    output [3:0] Operation
);
  
  wire [1:0] ALUop;
  
  Control_Unit C(
    .Opcode(Opcode),
    .Branch(Branch), 
    .MemRead(MemRead),
    .MemtoReg(MemtoReg),
    .ALUop(ALUop),
    .MemWrite(MemWrite),
    .ALUSrc(ALUSrc),
    .RegWrite(RegWrite)
  );
  
  ALU_Control A(
    .ALUop(ALUop),
    .Funct(Funct),
    .Operation(Operation)
  );
endmodule