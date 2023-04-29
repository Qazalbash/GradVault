"use strict";

let gl, points, colors, width, height;
let nt = 100,
    range = { start: -2, end: 2 };

window.onchange = () => {
    nt = Number(document.getElementById("nt").value);
    generate_set();
    render();
};
window.onload = () => {
    // fetching the html element
    let canvas = document.getElementById("gl-canvas");

    // getting the webgl2 context
    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL isn't available");

    // setting the viewing port and the default color of the canvas
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);

    width = canvas.width;
    height = canvas.height;

    // generating the points for mandelbrot set
    generate_set();

    render();
};

// square of the norm of a complex number
const normsq = (z) => z[0] * z[0] + z[1] * z[1];

// calculating if point in the mandelbrot set
const mandelbrot = (c) => {
    let z = vec2(0.0, 0.0),
        count = 0;

    do {
        z = vec2(z[0] * z[0] - z[1] * z[1] + c[0], 2.0 * z[0] * z[1] + c[1]);
        count++;
    } while (count < nt && normsq(z) <= 4.0);

    return count;
};

// function to generate the mandelbrot set
const generate_set = () => {
    colors = [];
    points = [];
    let c,
        z_,
        red = vec4(1.0, 0.0, 0.0, 1.0),
        green = vec4(0.0, 1.0, 0.0, 1.0),
        blue = vec4(0.0, 0.0, 1.0, 1.0);

    for (var a = 0; a <= width; a++) {
        for (var b = 0; b <= height; b++) {
            c = vec2(
                map_point(0, width, range.start, range.end, a),
                map_point(0, height, range.start, range.end, b)
            );

            z_ = vec2(
                map_point(range.start, range.end, -1, 1, c[0]),
                map_point(range.start, range.end, -1, 1, c[1])
            );

            points.push(z_);

            const escape_count = mandelbrot(c);

            if (escape_count < nt) {
                colors.push(map_point(0, nt / 5, red, green, escape_count));
            } else {
                colors.push(blue);
            }
        }
    }
};

const render = () => {
    let program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    // vertices
    let vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(points), gl.STATIC_DRAW);

    let vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    // colors
    let cBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(colors), gl.STATIC_DRAW);

    let vColor = gl.getAttribLocation(program, "vColor");
    gl.vertexAttribPointer(vColor, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vColor);

    // rendering
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, points.length);
};
