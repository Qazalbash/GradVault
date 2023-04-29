module MUX(
    input [63:0] A,
    input [63:0] B,
    output [63:0] O,
    input S
);

assign O = (S==1'b1) ? A: B;

endmodule




