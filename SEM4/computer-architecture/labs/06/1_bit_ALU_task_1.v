module MUX2x1
  (
    input A,
    input B,
    input S,
    output Y
  );

  assign Y = S ? B : A;
   
endmodule

module MUX4x1
  (
    input A,
    input B,
    input C,
    input D,
    input [1:0] S,
    output reg Y
  );

  always @*
    begin
      if (S==2'b00)
        Y = A;
      else if (S==2'b01)
        Y = B;
      else if (S==2'b10)
        Y = C;
      else
        Y = D;
    end
endmodule

module ALU_1_bit
  (
    input a,
    input b,
    input [3:0] ALUop,
    input cin,
    
    output cout,
    output result,
  );
  
  wire mux1out;
  wire mux2out;
  
  MUX2x1 I1(a, ~a, ALUop[3], mux1out);
  MUX2x1 I2(b, ~b, ALUop[2], mux2out);
  
  MUX4x1 I3(
    mux1out & mux2out,
    mux1out | mux2out,
    mux1out + mux2out + cin,
    0,
    ALUop[1:0],
    result
  );
  assign cout = (mux1out & cin) | (mux2out & cin) | (mux1out & mux2out);    

endmodule