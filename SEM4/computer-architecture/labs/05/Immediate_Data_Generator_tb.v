module tb();
  reg [31:0] dins;
  wire [63:0] did;
  
  Immediate_Data_Generator i1(
    .instruction(dins),
    .imm_data(did)
  );
  
  initial
    begin
      dins = 32'h02010513; // addi x10, x2, 32
      #10
      dins = 32'h00000263; // beq x0 x0 4
      #10
      dins = 32'h20002503; // lw x10, 0x200(x0)
      #10
      dins = 32'h02010513;
      $finish;
    end
  initial begin
    $dumpfile("dump.vcd");
    $dumpvars();
//     $monitor("INS = %h IMM = %d", dins, did);
  end
endmodule