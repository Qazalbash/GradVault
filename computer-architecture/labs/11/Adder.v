module Adder(
    input  [63:0] a,
    input  [63:0] b,
    output [63:0] c
);
    assign c = a + b;
    
endmodule