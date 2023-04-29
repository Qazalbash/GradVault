`include "branch_module.v"

`timescale 1ns/1ns

module tb();
    reg [2:0] funct3;
    reg  ZERO;
    reg  BRANCH;
    wire BEQ;
    wire BNE;

    branch_module T(
        .funct3(funct3),
        .ZERO(ZERO),
        .BRANCH(BRANCH),
        .BEQ(BEQ),
        .BNE(BNE)
    );

    initial begin
        #10
        ZERO=1'b0; BRANCH=1'b0; funct3 = 3'b000;
        #10
        ZERO=1'b0; BRANCH=1'b1; funct3 = 3'b000;
        #10
        ZERO=1'b1; BRANCH=1'b0; funct3 = 3'b000;
        #10
        ZERO=1'b1; BRANCH=1'b1; funct3 = 3'b000;
        #10
        ZERO=1'b0; BRANCH=1'b0; funct3 = 3'b001;
        #10
        ZERO=1'b0; BRANCH=1'b1; funct3 = 3'b001;
        #10
        ZERO=1'b1; BRANCH=1'b0; funct3 = 3'b001;
        #10
        ZERO=1'b1; BRANCH=1'b1; funct3 = 3'b001;
        #10
        ZERO=1'b0; BRANCH=1'b0; funct3 = 3'b011;
        #10
        ZERO=1'b0; BRANCH=1'b0; funct3 = 3'b011;
    end
    initial begin
        $dumpfile("testResults.vcd");
        $dumpvars();
    end
endmodule