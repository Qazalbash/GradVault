module registerFile (
    input  [63:0] WriteData,
    input  [4:0]  rs1,
    input  [4:0]  rs2,
    input  [4:0]  rd,
    input  RegWrite,
    input  clk,
    input  reset,
    output reg [63:0] ReadData1,
    output reg [63:0] ReadData2
);

    reg [63:0] registers [31:0];
    // Initialize Registers with some value
    initial begin
    registers[ 0] = 64'd0;  registers[ 1] = 64'd1;
    registers[ 2] = 64'd2;  registers[ 3] = 64'd3;
    registers[ 4] = 64'd4;  registers[ 5] = 64'd5;
    registers[ 6] = 64'd6;  registers[ 7] = 64'd7;
    registers[ 8] = 64'd8;  registers[ 9] = 64'd9;
    registers[10] = 64'd10; registers[11] = 64'd11;
    registers[12] = 64'd12; registers[13] = 64'd13;
    registers[14] = 64'd14; registers[15] = 64'd15;
    registers[16] = 64'd16; registers[17] = 64'd17;
    registers[18] = 64'd18; registers[19] = 64'd19;
    registers[20] = 64'd20; registers[21] = 64'd21;
    registers[22] = 64'd22; registers[23] = 64'd23;
    registers[24] = 64'd24; registers[25] = 64'd25;
    registers[26] = 64'd26; registers[27] = 64'd27;
    registers[28] = 64'd28; registers[29] = 64'd29;
    registers[30] = 64'd30; registers[31] = 64'd31; 
    end
    // Read Data
    always @(*) begin
        ReadData1 <= (reset == 1'b1) ? 0 : registers[rs1];
        ReadData2 <= (reset == 1'b1) ? 0 : registers[rs2];
    end

    // Write Data
    always @(posedge clk) begin
        if (RegWrite == 1'b1) begin
            registers[rd] = WriteData;
        end
    end

endmodule