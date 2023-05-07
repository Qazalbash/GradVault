module Program_Counter(
    input  clk, reset, 
    input  [63:0] PC_In,
    output reg [63:0] PC_Out
);
    always @(posedge clk or posedge reset)
        PC_Out = (reset==1'b1) ? 64'b0 : PC_In;

endmodule
