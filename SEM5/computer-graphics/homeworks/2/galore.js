"use strict";

var program;
var gl;
var cl;
var vertices = [];
var type = [];
var idx = 0;
var colors = [];
var click_count = 0;
var clicked_x;
var clicked_y;
var mode = 0;
var vs = 0;
var ts = 0;

function getMousePosition(canvas, event) {
    let rect = canvas.getBoundingClientRect();
    let x = event.clientX - rect.left;
    let y = event.clientY - rect.top;
    clicked_x = (x * 2) / canvas.width - 1;
    clicked_y = -((y * 2) / canvas.height - 1);
    click_count++;
    if ((vs == 0 && mode == 1) || (ts == 0 && mode == 0)) {
        cl = vec3(
            Math.random() % 0.9,
            Math.random() % 0.9,
            Math.random() % 0.9
        );
    }
    if (mode == 1) vs++;
    if (mode == 0) ts++;
    if (ts == 3 && mode == 0) ts = 0;
    if (vs == 4 && mode == 1) {
        vs = 0;

        var A1, A2, A3, B1, B2, B3, C1, C2, C3, x0, y0, m1, m2, m3, d1, d2, d3;
        x0 = clicked_x;
        y0 = clicked_y;
        m1 =
            (vertices[idx - 2][1] - vertices[idx - 1][1]) /
            (vertices[idx - 2][0] - vertices[idx - 1][0]);
        m2 =
            (vertices[idx - 2][1] - vertices[idx - 3][1]) /
            (vertices[idx - 2][0] - vertices[idx - 3][0]);
        m3 =
            (vertices[idx - 3][1] - vertices[idx - 1][1]) /
            (vertices[idx - 3][0] - vertices[idx - 1][0]);

        if (m1 == Infinity || m1 == -Infinity) {
            A1 = 0;
            B1 = 0;
            C1 = vertices[idx - 1][0];
        } else {
            A1 = -m1;
            B1 = 1;
            C1 = m1 * vertices[idx - 1][0] - vertices[idx - 1][1];
        }
        if (m2 == Infinity || m2 == -Infinity) {
            A2 = 0;
            B2 = 0;
            C2 = vertices[idx - 3][0];
        } else {
            B2 = 1;
            A2 = -m2;
            C2 = m2 * vertices[idx - 3][0] - vertices[idx - 3][1];
        }
        if (m3 == Infinity || m3 == -Infinity) {
            A3 = 0;
            B3 = 0;
            C3 = vertices[idx - 1][0];
        } else {
            B3 = 1;
            A3 = -m3;
            C3 = m3 * vertices[idx - 1][0] - vertices[idx - 1][1];
        }
        d1 =
            Math.abs(A1 * x0 + B1 * y0 + C1) /
            Math.sqrt(Math.pow(A1, 2) + Math.pow(B1, 2));
        d2 =
            Math.abs(A2 * x0 + B2 * y0 + C2) /
            Math.sqrt(Math.pow(A2, 2) + Math.pow(B2, 2));
        d3 =
            Math.abs(A3 * x0 + B3 * y0 + C3) /
            Math.sqrt(Math.pow(A3, 2) + Math.pow(B3, 2));

        if (d1 <= d2 && d1 <= d3) {
            vertices.push(vertices[idx - 2]);
            vertices.push(vertices[idx - 1]);
        } else if (d2 < d1 && d2 <= d3) {
            vertices.push(vertices[idx - 2]);
            vertices.push(vertices[idx - 3]);
        } else if (d3 < d1 && d3 < d2) {
            vertices.push(vertices[idx - 3]);
            vertices.push(vertices[idx - 1]);
        }

        vertices.push(vec3(clicked_x, clicked_y, 0.0));
        colors.push(cl);
        colors.push(cl);
        colors.push(cl);
        type.push(mode);
        type.push(mode);
        type.push(mode);
        idx += 2;
    } else {
        vertices.push(vec3(clicked_x, clicked_y, 0.0));
        colors.push(cl);
        type.push(mode);
    }
    idx++;
    render(vertices.length);
}

function key_pressed(n) {
    if (n == "r" || n == "R") {
        vertices = [];
        colors = [];
        type = [];
        mode = 0;
        idx = 0;
        vs = 0;
        ts = 0;
    } else if (n == "t" || n == "T") {
        if (mode == 0) set_mode_q();
        else set_mode_t();
    }
    render();
}

window.onload = () => {
    window.addEventListener(
        "keydown",
        function (e) {
            key_pressed(e.key);
        },
        false
    );
    let canvasElem = document.querySelector("canvas");

    canvasElem.addEventListener("mousedown", function (e) {
        getMousePosition(canvasElem, e);
    });

    let canvas = document.getElementById("gl-canvas");

    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL isn't available");

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    render();
};

function set_mode_q() {
    if (mode == 0) {
        var i = 0;
        for (var x = 0; x < type.length; x++) if (type[x] == 0) i++;

        var j = idx - 1;
        var k = i % 3;
        vs = k;
        while (k > 0) {
            type[j] = 1;
            k--;
            j--;
        }
    }
    mode = 1;
}

function set_mode_t() {
    if (mode == 1) {
        var i = 0;
        for (var x = 0; x < type.length; x++) if (type[x] == 1) i++;
        var j = idx - 1;
        var k = i % 6;
        ts = k;
        while (k > 0) {
            type[j] = 0;
            k--;
            j--;
        }
    }
    mode = 0;
}

let render = () => {
    if (vertices.length > 0) {
        let cBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, flatten(colors), gl.STATIC_DRAW);

        let vColor = gl.getAttribLocation(program, "vColor");
        gl.vertexAttribPointer(vColor, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(vColor);
        let vBuffer = gl.createBuffer();

        gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

        let vPosition = gl.getAttribLocation(program, "vPosition");
        gl.vertexAttribPointer(vPosition, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(vPosition);
    }

    gl.clear(gl.COLOR_BUFFER_BIT);
    var y = 0;
    var st = 0;
    while (y < vertices.length) {
        var current_mode = type[st];
        var sp = st + 1;
        while (type[sp] == current_mode) sp++;
        var tmp = sp - st;
        if (current_mode == 0 && tmp >= 3) {
            gl.drawArrays(gl.TRIANGLES, st, sp - ((sp - st) % 3) - st);
            gl.drawArrays(gl.POINTS, sp - ((sp - st) % 3), sp);
        } else if (current_mode == 1 && sp - st >= 6) {
            gl.drawArrays(gl.TRIANGLES, st, sp - ((sp - st) % 6) - st);
            gl.drawArrays(gl.POINTS, sp - ((sp - st) % 6), sp);
        } else gl.drawArrays(gl.POINTS, st, sp);

        st = sp;
        y = sp;
    }
};
