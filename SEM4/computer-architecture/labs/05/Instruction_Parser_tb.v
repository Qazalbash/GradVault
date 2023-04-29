module test_bench();
  reg [31:0] inst;
  wire [6:0] opcode;
  wire [4:0] rd;
  wire [2:0] func3;
  wire [4:0] rs1;
  wire [4:0] rs2;
  wire [6:0] func7;
  
  Instruction_Parser I
  (
    .instruction(inst),
    .opcode(opcode),
    .rd(rd),
    .func3(func3),
    .rs1(rs1),
    .rs2(rs2),
    .func7(func7)
  );
  
  initial
    begin
      inst = 32'h005201b3; // add x3, x4, x5
  	  #100
      inst = 32'h00820193; // addi x3, x4, 8
      #100
      inst = 32'h005201b3; // add x3, x4, x5
    end
  initial
    begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule