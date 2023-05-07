module tb();
  
  reg clk;
  reg reset;
  wire [31:0] instruction;
  
  Instruction_Fetch I(
    .clk(clk),
    .reset(reset),
    .Instruction(instruction)
  );
  
  always
    #10 clk = ~clk;
  
  initial begin
    clk = 1'b0;
    reset = 1'b1;
    #10
    reset = 1'b0;
  end
   
  initial begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule