module tb();
  reg [5:0] da;
  reg [5:0] db;
  wire dcout;
  reg dcin;
  reg [3:0] dop;
  wire [5:0] dr;
      
  ALU_6_bit I(
    .a(da),
    .b(db),
    .cout(dcout),
    .ALUop(dop),
    .result(dr),
    .cin(dcin)
  );
  
  	initial begin
        dop = 4'd0; da = 6'b101010; db = 6'b101100; dcin = 1'b1;
        #5
        dop = 4'd1; da = 6'b101010; db = 6'b101100; dcin = 1'b1;
        #5
        dop = 4'd2; da = 6'b101010; db = 6'b101100; dcin = 1'b1;
        #5
        dop = 4'd6; da = 6'b101010; db = 6'b101100; dcin = 1'b1;
        #5
        dop = 4'd12; da = 6'b101010; db = 6'b101100; dcin = 1'b1;
        #5
        dop = 4'd2; // Dummy Case
        $finish;
    end
  
  initial begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule		