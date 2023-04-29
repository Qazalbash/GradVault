//~ `New testbench
`include "registerfile.v"
`timescale  1ps / 1ps

module tb_registerFile();

// registerFile Inputs
reg [63:0] WriteData;
reg [4:0] RS1;
reg [4:0] RS2;
reg [4:0] RD;
reg RegWrite;
reg clk;
reg reset;

// registerFile Outputs
wire [63:0] ReadData1;
wire [63:0] ReadData2;


initial begin
    clk=1'b0;
end

always
    #5
    clk=~clk;

registerFile  u_registerFile (
    .WriteData(WriteData[63:0]),
    .RS1(RS1[4:0]),
    .RS2(RS2[4:0]),
    .RD(RD[4:0]),
    .RegWrite(RegWrite),
    .clk(clk),
    .reset(reset),

    .ReadData1(ReadData1[63:0]),
    .ReadData2(ReadData2[63:0])
);

initial begin
    WriteData = 64'd30; RS1 = 64'd30; RS2 = 64'd29; RD = 64'd30; RegWrite = 1'b1; reset = 1'b0;
    #6
    WriteData = 64'h300; RS1 = 64'd30; RS2 = 64'd29; RD = 64'd30; RegWrite = 1'b1;
    #6
    RegWrite = 1'b0; 
    #6
    WriteData = 64'd31; RegWrite = 1'b1; 
    #6
    WriteData = 64'd30; RS1 = 64'd28; RS2 = 64'd29; RD = 64'd30; RegWrite = 1'b1; 
    $finish;
end

initial begin
    $dumpfile("registerfilewave.vcd");
    $dumpvars();
end
endmodule
