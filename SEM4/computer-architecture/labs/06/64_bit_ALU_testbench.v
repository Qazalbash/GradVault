module tb();
  reg [63:0] da;
  reg [63:0] db;
  wire z;
  
  reg [3:0] dop;
  wire [63:0] dr;
      
  ALU_64_bit I(
    .a(da),
    .b(db),
    .zero(z),
    .ALUop(dop),
    .result(dr)
  );
  
  	initial begin
        dop = 4'd0; da = 64'd03351120129; db = 64'd080080800;
        #5
        dop = 4'd1; da = 64'd03351120129; db = 64'd080080800;
        #5
        dop = 4'd2; da = 64'd03351120129; db = 64'd080080800;
        #5
        dop = 4'd6; da = 64'd03351120129; db = 64'd080080800;
        #5
        dop = 4'd12; da = 64'd03351120129; db = 64'd080080800;
        #5
        dop = 4'd2; // Dummy Case
        $finish;
    end
  
  initial begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule		