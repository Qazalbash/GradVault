`include "registerFile.v"
`include "instruction_parser.v"

module toplevel(
    input [31:0] instruction,
    output [63:0] ReadData1,
    output [63:0] ReadData2
); 
    wire [6:0] opcode;
    wire [4:0] rd;
    wire [2:0] funct3;
    wire [4:0] rs1;
    wire [4:0] rs2;
    wire [6:0] funct7;
    wire [63:0] WriteData;
    wire clk, reset, RegWrite;

    assign reset = 1'b0;
    assign RegWrite = 1'b0;

    Instruction_Parser i1(
        .instruction(instruction), 
        .opcode(opcode), 
        .rd(rd), 
        .rs1(rs1), 
        .rs2(rs2), 
        .funct3(funct3), 
        .funct7(funct7)
    );

    registerFile r1(
        .WriteData(WriteData), 
        .RS1(rs1), 
        .RS2(rs2), 
        .RD(rd), 
        .clk(clk), 
        .reset(reset), 
        .RegWrite(RegWrite), 
        .ReadData1(ReadData1), 
        .ReadData2(ReadData2)
    );
endmodule