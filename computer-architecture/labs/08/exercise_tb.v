module tb();
    reg  [63:0] dMA;
    reg  [63:0] dWD;
    reg         dc;
    reg         dMW;
    reg         dMR;
    wire [63:0] dRD;

    initial
        begin
            dc = 1;
            
            dMA = 64'b000; dMW = 1; dMR = 1;
            
            #10 dMA = 64'b100; dMW = 0; dMR = 1;
            
            #10 dMA = 64'b000; dMW = 1; dMR = 1;
            
            #10 dMA = 64'b100; dMW = 0; dMR = 0;

            #100 $finish;
        end
    
    
    always #5
        dc = ~dc;
    
    Data_Memory g1(
        dMA,
        dWD, 
        dc, 
        dMW, 
        dMR, 
        dRD
    );
    
    initial
        begin
        $dumpfile("testResults.vcd");
        $dumpvars(
            1,
            tb
        );
        $monitor(
            "Time=%0d, Mem_Addr=%64b. Write_Data=%64b. clk=%1b .MemWrite=%1b .MemRead=%1b .Read_Data=%64b\n",
            $time,
            dMA, 
            dWD, 
            dc, 
            dMW, 
            dMR, 
            dRD
        );
        end
endmodule