module tb();
  reg da;
  reg db;
  wire dcout;
  reg dcin;
  reg [3:0] dop;
  wire dr;
      
  ALU_1_bit I(
    .a(da),
    .b(db),
    .cout(dcout),
    .ALUop(dop),
    .result(dr),
    .cin(dcin)
  );
  
  	initial begin
        dop = 4'd0; da = 1'b1; db = 1'b0; dcin = 1'b1;
        #5
        dop = 4'd1; da = 1'b1; db = 1'b0; dcin = 1'b1;
        #5
        dop = 4'd2; da = 1'b1; db = 1'b0; dcin = 1'b1;
        #5
        dop = 4'd6; da = 1'b1; db = 1'b0; dcin = 1'b1;
        #5
        dop = 4'd12; da = 1'b1; db = 1'b0; dcin = 1'b1;
        #5
        dop = 4'd2; // Dummy Case
        $finish;
    end
  
  initial begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule		