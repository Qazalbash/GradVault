"use strict";

let gl,
    vertices,
    colors,
    duo_first = {
        top: vec2(-1.0, 1.0),
        bottom: vec2(-1.0, 0.0),
        color: vec4(0.0, 0.0, 0.0, 1.0),
    },
    duo_second = {
        top: vec2(1.0, 1.0),
        bottom: vec2(1.0, 0.0),
        color: vec4(1.0, 1.0, 1.0, 1.0),
    },
    trio_first = {
        top: vec2(-1.0, 0.0),
        bottom: vec2(-1.0, -1.0),
        color: vec4(1.0, 0.0, 0.0, 1.0),
    },
    trio_second = {
        top: vec2(0.0, 0.0),
        bottom: vec2(0.0, -1.0),
        color: vec4(0.0, 1.0, 0.0, 1.0),
    },
    trio_third = {
        top: vec2(1.0, 0.0),
        bottom: vec2(1.0, -1.0),
        color: vec4(0.0, 0.0, 1.0, 1.0),
    };

// event to check the color has changed

const hexToRgb = (hex) => {
    let r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
    return vec4(r / 255, g / 255, b / 255, 1.0);
};

window.onchange = () => {
    let e1 = document.getElementById("duo-color-first"),
        e2 = document.getElementById("duo-color-second"),
        e3 = document.getElementById("trio-color-first"),
        e4 = document.getElementById("trio-color-second"),
        e5 = document.getElementById("trio-color-third");

    duo_first.color = hexToRgb(e1.value);
    duo_second.color = hexToRgb(e2.value);
    trio_first.color = hexToRgb(e3.value);
    trio_second.color = hexToRgb(e4.value);
    trio_third.color = hexToRgb(e5.value);

    const width = gl.canvas.width;

    colors = [];
    vertices = [];

    strip(width, duo_first, duo_second);
    strip(width, trio_first, trio_second, trio_third);

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

    // width of the canvas
    const width = canvas.width;

    colors = [];
    vertices = [];

    // for grey scale strip
    strip(width, duo_first, duo_second);

    // for rgb strip
    strip(width, trio_first, trio_second, trio_third);

    // rendering the lines
    render();
};

const strip = (width, ...lines) => {
    let interpolated_colors;
    width = width / (lines.length - 1);
    for (var i = 0; i + 1 < lines.length; i++) {
        interpolated_colors = color_interpolation(lines[i], lines[i + 1], width);
        vertices = vertices.concat(interpolated_colors.point);
        colors = colors.concat(interpolated_colors.color);
    }
};

let color_interpolation = (left, right, width) => {
    let colors = [],
        points = [],
        line_color;
    for (let i = 0; i <= width; i++) {
        // interpolating the color

        line_color = map_point(left.color, right.color, left.color, right.color, i / width);

        /*
        Adding two times because we are rendering the line primitives.
        Therefore we have one color for the top of the line and one 
        for the bottom of the line.
        */
        colors.push(line_color, line_color);

        // interpolating the top and bottom vertices of the lines
        let top = mix(left.top, right.top, i / width),
            bottom = mix(left.bottom, right.bottom, i / width);

        points.push(top, bottom);
    }
    return {
        point: points,
        color: colors,
    };
};

let render = () => {
    let program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    // vertices
    let vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

    let vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    // color of vertices
    let cBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(colors), gl.STATIC_DRAW);

    let vColor = gl.getAttribLocation(program, "vColor");
    gl.vertexAttribPointer(vColor, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vColor);

    // rendering
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.LINES, 0, vertices.length);
};
