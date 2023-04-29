// Code your testbench here
// or browse Examples
module MUX_tb;
  reg [63:0] dA ;
  reg [63:0] dB;
  reg dS ;
  wire [63:0] dY; 
  
  MUX g1(dA,dB,dS,dY);
 
  initial 
    begin
      dA = 64'd1236; dB = 64'd44865; dS = 1'b0;
      #100
      dA = 64'd12376; dB = 64'd4865; dS = 1'b1;
      #100
      dA = 64'd1236; dB = 64'd44865; dS = 1'b0;
    end
 
  initial
    begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule
