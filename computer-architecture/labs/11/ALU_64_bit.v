module ALU_64_bit(
    input  [63:0] A,
    input  [63:0] B,
    input  [ 3:0] Operation, 
    output reg [63:0] O, 
    output reg Zero
);

    always@(*)
        begin
            case (Operation)
                4'd0:     O <= A & B; 
                4'd1:     O <= A | B; 
                4'd2:     O <= A + B; 
                4'd6:     O <= A - B; 
                default:  O <=~(A|B);
            endcase
            Zero = (O == 64'b0) ? 1'b1 : 1'b0;
        end
    
endmodule
