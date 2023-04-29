module tb();
 
    reg  [63:0] dA;
    wire [31:0] dI;
    
    initial
        begin
            dA = 64'b00000;
            
            #10 dA = 64'b00100;
            
            #10 dA = 64'b01000;
            
            #10 dA = 64'b01100;
            
            #10 dA = 64'b10000;
            
            #10 dA = 64'b1000;
        
            #10 dA = 64'b00000; //dummy
        end
    
    Instruction_Memory g1(
        dA,
        dI
    );
    
    initial
        begin
            $dumpfile("testResults.vcd");
            $dumpvars(
                1,
                tb
            );
            $monitor(
                "Time=%0d,
                Inst_Address=%64b. Instruction=%32b\n",
                $time,
                dA,
                dI
            );
        end
endmodule