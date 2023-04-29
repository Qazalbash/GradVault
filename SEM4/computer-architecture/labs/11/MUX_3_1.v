module MUX_3_1(
    input [63:0] A,
    input [63:0] B,
    input [63:0] C,
    output [63:0] O,
    input [1:0] S
);

assign O = (S==2'b00) ? A: (S==2'b01) ? B: C;

endmodule