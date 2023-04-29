module tb();
  
    reg [6:0] Opcode;
    reg [3:0] Funct;
    wire Branch ;
    wire MemRead;
    wire MemtoReg;
    wire MemWrite;
    wire ALUSrc;
    wire RegWrite;
  	wire [3:0] Operation;
  
  Top_Control T(
    .Opcode(Opcode),
    .Funct(Funct),
    .Branch(Branch), 
    .MemRead(MemRead),
    .MemtoReg(MemtoReg),
    .MemWrite(MemWrite),
    .ALUSrc(ALUSrc), 
    .RegWrite(RegWrite),
    .Operation(Operation)
);
  
  
  initial begin
    #10
    Opcode = 7'b0110011 ; Funct = 4'b0000;
    #10
    Opcode = 7'b0110011 ; Funct = 4'b1000;
    #10
    Opcode = 7'b0000011 ; Funct = 4'b0111;
    #10
    Opcode = 7'b0000011 ; Funct = 4'b0110;
    #10
    Opcode = 7'b0100011 ; Funct = 4'b0000;
    #10
    Opcode = 7'b0100011 ; Funct = 4'b1000;
    #10
    Opcode = 7'b1100011 ; Funct = 4'b0111;
    #10
    Opcode = 7'b1100011 ; Funct = 4'b0110;
    #10
    Opcode = 7'b1100011 ; Funct = 4'b0110;
    #10
    Opcode = 7'b0010011 ; Funct = 4'b0100;
    #10
    Opcode = 7'b0010011 ; Funct = 4'b0110;
  end
   
  initial begin
      $dumpfile("testResults.vcd");
      $dumpvars();
    end
endmodule