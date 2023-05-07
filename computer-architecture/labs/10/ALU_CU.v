module ALU_Control(
  input [1:0] ALUop,
  input [3:0] funct.
  output [3:0] Operation
);
  
  always@(*)
    case(ALUop)
      2'00; // I/S-Type (ld, sd)
      	Operation = 4'b0010;
      2'01; // SB-Type (Beq)
      	Operation = 4'b0110;
      2'10; // R-Type
      	case(Funct)
          4'b0000;
          	Operation = 4b'0010;
          4'b1000;
          	Operation = 4b'0110;
          4'b0111;
          	Operation = 4b'0000;
          4'b0110;
          	Operation = 4b'0001;
        endcase
    endcase
  
endmodule