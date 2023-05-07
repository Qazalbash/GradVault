module  Immediate_Data_Generator
  (
    input [31:0] instruction,
    output reg [63:0] imm_data
  );
  
  always @instruction
    begin
      if (instruction[6:5] == 2'b00) // I type
        begin
          imm_data[11:0]=instruction[31:20];
        end

      else if (instruction[6:5] == 2'b01) // S type
        begin
          imm_data[4:0]=instruction[11:7];
          imm_data[11:5]=instruction[31:25];
        end

      else if (instruction[6:5] == 2'b11) // B type
        begin
          imm_data[10]=instruction[7];
          imm_data[11]=instruction[31];
          imm_data[3:0]=instruction[10:8];
          imm_data[9:4]=instruction[30:25];
        end

      imm_data[63:12] = {52{imm_data[11]}};
    end

endmodule