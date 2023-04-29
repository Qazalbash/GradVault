`include "MUX_3_1.v"

module immediate_generator(
    input [31:0] instruction,
    output wire [63:0] immed_value
);

// imm[6:5] == 00 for load instr       
// imm[6:5] == 01 for store instr      
// imm[6:5] == 1X for conditional instr

MUX_3_1 m1(
        .A({{52{instruction[31]}},instruction[31:20]}),
        .B({{52{instruction[31]}},instruction[31:25], instruction[11:7]}),
        .C({{52{instruction[31]}},instruction[31], instruction[7], instruction[30:25], instruction[11:8]}),
        .O(immed_value), .S(instruction[6:5])
    );
    
endmodule
