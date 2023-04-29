module Data_Memory(
    input  [63:0] Mem_Addr,
    input  [63:0] WriteData,
    input  clk, MemWrite, MemRead,
    output reg [63:0] Read_Data
);

    reg [7:0] data_mem [63:0];
//  data_mem initialization
    initial
        begin
            data_mem[ 0] = 8'd0; data_mem[ 1] = 8'd0; data_mem[ 2] = 8'd0; data_mem[ 3] = 8'd0; data_mem[ 4] = 8'd0;
            data_mem[ 5] = 8'd0; data_mem[ 6] = 8'd0; data_mem[ 7] = 8'd0; data_mem[ 8] = 8'd0; data_mem[ 9] = 8'd0;
            data_mem[10] = 8'd1; data_mem[11] = 8'd0; data_mem[12] = 8'd0; data_mem[13] = 8'd0; data_mem[14] = 8'd0;
            data_mem[15] = 8'd0; data_mem[16] = 8'd0; data_mem[17] = 8'd0; data_mem[18] = 8'd0; data_mem[19] = 8'd0;
            data_mem[20] = 8'd2; data_mem[21] = 8'd0; data_mem[22] = 8'd0; data_mem[23] = 8'd0; data_mem[24] = 8'd0;
            data_mem[25] = 8'd0; data_mem[26] = 8'd0; data_mem[27] = 8'd0; data_mem[28] = 8'd0; data_mem[29] = 8'd0;
            data_mem[30] = 8'd3; data_mem[31] = 8'd0; data_mem[32] = 8'd0; data_mem[33] = 8'd0; data_mem[34] = 8'd0;
            data_mem[35] = 8'd0; data_mem[36] = 8'd0; data_mem[37] = 8'd0; data_mem[38] = 8'd0; data_mem[39] = 8'd0;
            data_mem[40] = 8'd4; data_mem[41] = 8'd0; data_mem[42] = 8'd0; data_mem[43] = 8'd0; data_mem[44] = 8'd0;
            data_mem[45] = 8'd0; data_mem[46] = 8'd0; data_mem[47] = 8'd0; data_mem[48] = 8'd0; data_mem[49] = 8'd0;
            data_mem[50] = 8'd5; data_mem[51] = 8'd0; data_mem[52] = 8'd0; data_mem[53] = 8'd0; data_mem[54] = 8'd0;
            data_mem[55] = 8'd0; data_mem[56] = 8'd0; data_mem[57] = 8'd0; data_mem[58] = 8'd0; data_mem[59] = 8'd0;
            data_mem[60] = 8'd6; data_mem[61] = 8'd0; data_mem[62] = 8'd0; data_mem[63] = 8'd0;
        end

    always @(posedge clk)
    begin
        if(MemWrite == 1'b1)
            begin
                data_mem[Mem_Addr+7] = WriteData[63:56];
                data_mem[Mem_Addr+6] = WriteData[55:48];
                data_mem[Mem_Addr+5] = WriteData[47:40];
                data_mem[Mem_Addr+4] = WriteData[39:32];
                data_mem[Mem_Addr+3] = WriteData[31:24];
                data_mem[Mem_Addr+2] = WriteData[23:16];
                data_mem[Mem_Addr+1] = WriteData[15: 8];
                data_mem[Mem_Addr  ] = WriteData[ 7: 0];
            end
    end

    always @(Mem_Addr, MemRead)
    begin
        if (MemRead == 1'b1) begin
            Read_Data <= {data_mem[Mem_Addr+7], data_mem[Mem_Addr+6], data_mem[Mem_Addr+5], data_mem[Mem_Addr+4], data_mem[Mem_Addr+2], data_mem[Mem_Addr+1], data_mem[Mem_Addr+0]};
        end
    end

endmodule