module branch_module (
    input [2:0] funct3,
    input ZERO,
    input BRANCH,
    output reg BEQ,
    output reg BNE
);

    always @(*)
    begin
        if(BRANCH == 1'b1)
            begin
                if (funct3 == 3'b000) // BEQ
                    begin
                        if (ZERO == 1'b1)
                            BEQ = 1'b1;
                        else
                            BEQ = 1'b0;    
                        BNE = 1'b0;
                    end
                else if (funct3 == 3'b001) // BNE
                    begin
                        if (ZERO == 1'b1)
                            BNE = 1'b0;
                        else
                            BNE = 1'b1;
                        BEQ = 1'b0;    
                    end
            end
        else
            begin
                BEQ = 1'bx;
                BNE = 1'bx;
            end
    end

endmodule