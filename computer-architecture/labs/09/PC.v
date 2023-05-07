module Program_Counter(
	input clk,
  	input reset,
  	input [63:0] PC_in,
  	output reg [63:0] PC_out
);
  always@(posedge clk or posedge reset)
    begin
      if (reset == 1'b1)
        PC_out = 0;
      else
        PC_out = PC_in;
    end
endmodule