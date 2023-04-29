module registerFile(
    input [63:0] WriteData,
    input [4:0] RS1,
    input [4:0] RS2,
    input [4:0] RD,
    input RegWrite,
    
    input clk,
    input reset,
    
    output reg [63:0] ReadData1,
    output reg [63:0] ReadData2
);
  
    reg [31:0] Registers [63:0];
    initial
        begin
            Registers[ 0] = 64'd00000000; Registers[ 1] = 64'd71388834;
            Registers[ 2] = 64'd97631452; Registers[ 3] = 64'd62938961;
            Registers[ 4] = 64'd62600394; Registers[ 5] = 64'd71367855;
            Registers[ 6] = 64'd14065990; Registers[ 7] = 64'd14908608;
            Registers[ 8] = 64'd73652147; Registers[ 9] = 64'd13975018;
            Registers[10] = 64'd11621942; Registers[11] = 64'd20711349;
            Registers[12] = 64'd34563778; Registers[13] = 64'd73909590;
            Registers[14] = 64'd14124113; Registers[15] = 64'd35121869;
            Registers[16] = 64'd17697951; Registers[17] = 64'd17838443;
            Registers[18] = 64'd13507376; Registers[19] = 64'd15996910;
            Registers[20] = 64'd15551366; Registers[21] = 64'd10526495;
            Registers[22] = 64'd18722254; Registers[23] = 64'd28749826;
            Registers[24] = 64'd74111082; Registers[25] = 64'd14853443;
            Registers[26] = 64'd24186719; Registers[27] = 64'd11312423;
            Registers[28] = 64'd80539398; Registers[29] = 64'd18332851;
            Registers[30] = 64'd16196138; Registers[31] = 64'd10972098;
        end
  
    always@(*)
        begin
            ReadData1 = Registers[RS1];
            ReadData2 = Registers[RS2];
        end
  
    always@(posedge clk)
        begin
            if(RegWrite == 1'b1 && RD != 5'b00000)
                Registers[RD] = WriteData;
            end
  
endmodule