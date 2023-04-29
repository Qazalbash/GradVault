module Data_Memory(
    input      [63:0] Mem_Addr,
    input      [63:0] Write_Data,
    input             clk,
    input             MemWrite,
    input             MemRead,
    output reg [63:0] Read_Data
);
  
  reg [7:0] A [63:0];
  
  initial
    begin
        A[ 0] =  0; A[ 1] =  1; A[ 2] =  2;
        A[ 3] =  3; A[ 4] =  4; A[ 5] =  5;
        A[ 6] =  6; A[ 7] =  7; A[ 8] =  8;
        A[ 9] =  9; A[10] = 10; A[11] = 11;
        A[12] = 12; A[13] = 13; A[14] = 14;
        A[15] = 15; A[16] = 16; A[17] = 17;
        A[18] = 18; A[19] = 19; A[20] = 20;
        A[21] = 21; A[22] = 22; A[23] = 23;
        A[24] = 24; A[25] = 25; A[26] = 26;
        A[27] = 27; A[28] = 28; A[29] = 29;
        A[30] = 30; A[31] = 31; A[32] = 32;
        A[33] = 33; A[34] = 34; A[35] = 35;
        A[36] = 36; A[37] = 37; A[38] = 38;
        A[39] = 39; A[40] = 40; A[41] = 41;
        A[42] = 42; A[43] = 43; A[44] = 44;
        A[45] = 45; A[46] = 46; A[47] = 47;
        A[48] = 48; A[49] = 49; A[50] = 50;
        A[51] = 51; A[52] = 52; A[53] = 53;
        A[54] = 54; A[55] = 55; A[56] = 56;
        A[57] = 57; A[58] = 58; A[59] = 59;
        A[60] = 60; A[61] = 61; A[62] = 62;
        A[63] = 63;
    end   
    
    always @(*)
        if (MemRead == 1)
            begin
                Read_Data [07:00] = A [Mem_Addr + 0];
                Read_Data [15:08] = A [Mem_Addr + 1];
                Read_Data [23:16] = A [Mem_Addr + 2];
                Read_Data [31:24] = A [Mem_Addr + 3];
                Read_Data [39:32] = A [Mem_Addr + 4];
                Read_Data [47:40] = A [Mem_Addr + 5];
                Read_Data [55:48] = A [Mem_Addr + 6];
                Read_Data [63:56] = A [Mem_Addr + 7];
            end

  
    always @ (posedge clk)
        if (MemWrite == 1)
            begin
                A[Mem_Addr + 0] = Write_Data [07:00];
                A[Mem_Addr + 1] = Write_Data [15:08];
                A[Mem_Addr + 2] = Write_Data [23:16];
                A[Mem_Addr + 3] = Write_Data [31:24];
                A[Mem_Addr + 4] = Write_Data [39:32];
                A[Mem_Addr + 5] = Write_Data [47:40];
                A[Mem_Addr + 6] = Write_Data [55:48];
                A[Mem_Addr + 7] = Write_Data [63:56];
            end
endmodule
  