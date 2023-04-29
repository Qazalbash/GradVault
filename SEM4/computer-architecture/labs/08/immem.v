module Instruction_Memory(
    input      [63:0] Inst_Address,
    output reg [31:0] Instruction
);
  
    reg [7:0] I[15:0];
  
    initial 
        begin
            I[ 0] = 8'b00000000; I[ 1] = 8'b00000001;
            I[ 2] = 8'b00000010; I[ 3] = 8'b00000011;
            I[ 4] = 8'b00000100; I[ 5] = 8'b00000101;
            I[ 6] = 8'b00000110; I[ 7] = 8'b00000111;
            I[ 8] = 8'b00001000; I[ 9] = 8'b00001001;
            I[10] = 8'b00001010; I[11] = 8'b00001011;
            I[12] = 8'b00001100; I[13] = 8'b00001101;
            I[14] = 8'b00001110; I[15] = 8'b00001111;
        end
    

    always @(*)
        begin
            Instruction [7:0]   = I[Inst_Address + 0];  // [3:0]
            Instruction [15:8]  = I[Inst_Address + 1];  // [7:4]
            Instruction [23:16] = I[Inst_Address + 2];  // [11:8]
            Instruction [31:24] = I[Inst_Address + 3];  // [15:12]
        end
endmodule
  