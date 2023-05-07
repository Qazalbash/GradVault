`include "RISC_V_Processor.v"

`timescale 1ns/1ns

module RISC_V_Pr_tb(); 
    reg clk, reset;
  
    RISC_V_Processor RISCV(.clk(clk), .reset(reset));

    initial 
        begin
        clk = 1'b0;
        reset = 1'b1;
        #10
        reset = 1'b0;
        end
  
    always  #5 clk = ~clk; 

    initial begin
      $dumpfile("RISC_V_Processor.vcd");
      $dumpvars();
      #100 $finish;
    end

endmodule
