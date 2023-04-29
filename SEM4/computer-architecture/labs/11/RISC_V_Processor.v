// Include statements
`include "immediate_generator.v"
`include "instruction_Memory.v"
`include "instruction_parser.v"
`include "Program_Counter.v"
`include "registerFile.v"
`include "Control_Unit.v"
`include "Data_Memory.v"
`include "ALU_Control.v"
`include "ALU_64_bit.v"
`include "Adder.v"
`include "MUX.v"

// Top-lelvel module 
module RISC_V_Processor(
    input clk, reset
);
    wire [63: 0] PC_Out, ReadData1, ReadData2, WriteData, ALUresult, imm_data, ReadDataMem;
    wire [63: 0] Adder1Out, Adder2Out, MuxBranchOut, MuxALUOut, MuxMemOut;
    wire [31: 0] instruction;
    wire [ 6: 0] Opcode, funct7;
    wire [ 4: 0] rs1, rs2, rd;
    wire [ 3: 0] Funct, Operation;
    wire [ 2: 0] funct3;
    wire [ 1: 0] ALUOp;
    wire Branch, MemRead, MemtoReg, MemWrite, ALUSrc, RegWrite;

    Program_Counter PC(
            .clk(clk),
            .reset(reset),
            .PC_In(MuxBranchOut),
            .PC_Out(PC_Out)
        );
    
    immediate_generator Igen(
            .instruction(instruction),
            .immed_value(imm_data)
        );
    
    Adder add1(
            .a(64'd4),
            .b(PC_Out),
            .c(Adder1Out)
        );
    Adder add2(
            .a(PC_Out),
            .b(imm_data << 1),
            .c(Adder2Out)
        );
    
    MUX muxBranch(
            .A(Adder2Out),
            .B(Adder1Out),
            .O(MuxBranchOut),
            .S(Branch&Zero)
        );
    
    Instruction_Memory iMem(
            .Inst_address(PC_Out),
            .Instruction(instruction)
        );

    Control_Unit c1(
            .Opcode(Opcode),
            .Branch(Branch), 
            .MemRead(MemRead), 
            .MemtoReg(MemtoReg), 
            .MemWrite(MemWrite), 
            .ALUSrc(ALUSrc), 
            .RegWrite(RegWrite), 
            .ALUOp(ALUOp)
        );
    
    instruction_parser iParser(
            .instruction(instruction),
            .opcode(Opcode),
            .rd(rd),
            .rs1(rs1),
            .rs2(rs2),
            .funct3(funct3),
            .funct7(funct7)
        );
    
    registerFile rFile(
            .WriteData(MuxMemOut), 
            .rs1(rs1), 
            .rs2(rs2), 
            .rd(rd), 
            .clk(clk), 
            .reset(reset), 
            .RegWrite(RegWrite), 
            .ReadData1(ReadData1), .ReadData2(ReadData2)
        );
    
    ALU_Control ac1(
            .ALUOp(ALUOp),
            .Funct({instruction[30], funct3}),
            .Operation(Operation)
        );
    
    MUX muxALUSrc(
            .B(ReadData2),
            .A(imm_data),
            .O(MuxALUOut),
            .S(ALUSrc)
        );

    ALU_64_bit ALU64(
            .A(ReadData1),
            .B(MuxALUOut), 
            .O(ALUresult), 
            .Zero(Zero), 
            .Operation(Operation)
        );
    
    Data_Memory DMem(
            .Mem_Addr(ALUresult),
            .WriteData(ReadData2),
            .MemWrite(MemWrite), 
            .MemRead(MemRead), 
            .clk(clk), 
            .Read_Data(ReadDataMem)
        );

    MUX muxMemory(
            .A(ReadDataMem),
            .B(ALUresult),
            .O(MuxMemOut),
            .S(MemtoReg)
        );

endmodule
