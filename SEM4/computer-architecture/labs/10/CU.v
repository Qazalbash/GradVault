module Control_Unit(
    input [6:0] Opcode,
    output Branch, 
    output MemRead,
    output MemtoReg,
    output [1:0] ALUop,
    output MemWrite,
    output ALUSrc,
    output RegWrite
);
  always@(*)
    case(Opcode)

      7'b0110011; // R-Type
      	begin
          ALUSrc = 1b'0;
          MemtoReg =1b'0;
          RegWrite = 1b'1;
          MemRead = 1b'0;
          MemWrite = 1b'0;
          Branch = 1b'0;
          ALUOp = 2b'10;
        end
      7'b0000011 // I-Type (ld)
      	begin
          ALUSrc = 1b'1;
          MemtoReg =1b'1;
          RegWrite = 1b'1;
          MemRead = 1b'1;
          MemWrite = 1b'0;
          Branch = 1b'0;
          ALUOp = 2b'00;
        end
      7'b0100011 // I-Type (sd)
        begin
          ALUSrc = 1b'1;
          RegWrite = 1b'0;
          MemRead = 1b'0;
          MemWrite = 1b'1;
          Branch = 1b'0;
          ALUOp = 2b'00;
        end
      7'b1100011 // SB-Type (Beq)
        begin
          ALUSrc = 1b'0;
          RegWrite = 1b'0;
          MemRead = 1b'0;
          MemWrite = 1b'0;
          Branch = 1b'1;
          ALUOp = 2b'01;
         end


    endcase
  
endmodule