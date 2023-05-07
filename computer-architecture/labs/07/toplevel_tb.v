module toplevel_tb();
    reg [31:0] instruction;
    wire [63:0] ReadData1;
    wire [63:0] ReadData2;

    toplevel t1(
      .instruction(instruction),
      .ReadData1(ReadData1),
      .ReadData2(ReadData2)
    );

    initial begin
        instruction = 32'h001101b3;     // add x3, x2, x1
        #5
        instruction = 32'h016a8a33;     // add x20, x21, x22
        #5
        instruction = 32'h01ce8f33;     // add 30, 29, 28
        #5
        instruction = 32'd41474834;     // Redundant instruction
        $finish;
    end

    initial begin
        $dumpfile("toplevel.vcd");
        $dumpvars();
    end


endmodule
