"use strict";

let gl; // WebGL context
let vertices = []; // array of all vertices
let colors = []; // array of all colors
let shift = -33; // initial shift of pacman
let mouth_open = true; // is pacman mouth open
let count = 0; // counter for mouth opening

const speed = 0.5; // speed of pacman
const DELTA = 0.04; // distance between two points
const RED = vec4(1.0, 0.0, 0.0, 1.0); // red color
const GREEN = vec4(0.0, 1.0, 0.0, 1.0); // green color
const BLUE = vec4(0.0, 0.0, 1.0, 1.0); // blue color

window.onload = () => {
    let canvas = document.getElementById("gl-canvas");
    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL 2.0 isn't available");

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0923341336686, 0.0923009263738, 0.0923433085636, 1.0);

    render();
};

/**
 * all the points between xmin, xmax at the height of y
 * @param {Number} xmin
 * @param {Number} xmax
 * @param {Number} y
 * @returns {Array} array of points
 */
const strech = (xmin, xmax, y) => {
    return Array.from({ length: xmax - xmin + 1 }, (v, i) => {
        return vec2(i + xmin, y);
    });
};

/**
 * render the scene
 */
const render = () => {
    vertices = [];
    colors = [];

    const ghostBLUE = GHOST(shift - 75, 0, BLUE);
    const ghostGREEN = GHOST(shift - 50, 0, GREEN);
    const ghostRED = GHOST(shift - 25, 0, RED);
    const pacman = PACMAN(shift, 0, mouth_open);

    const seeds = [
        vec2(-20, 0),
        vec2(-15, 0),
        vec2(-10, 0),
        vec2(-5, 0),
        vec2(0, 0),
        vec2(5, 0),
        vec2(10, 0),
        vec2(15, 0),
        vec2(20, 0),
    ];

    for (var i = 0; i < seeds.length; ++i) {
        if (seeds[i][0] > 5 + shift) {
            vertices.push(seeds[i]);
        }
    }

    colors = Array.from({ length: vertices.length }, (v, i) =>
        vec4(1.0, 0.2, 0.1, 1.0)
    );

    vertices = vertices.concat(pacman.vertices);
    vertices = vertices.concat(ghostRED.vertices);
    vertices = vertices.concat(ghostGREEN.vertices);
    vertices = vertices.concat(ghostBLUE.vertices);

    colors = colors.concat(pacman.colors);
    colors = colors.concat(ghostRED.colors);
    colors = colors.concat(ghostGREEN.colors);
    colors = colors.concat(ghostBLUE.colors);

    let program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

    let vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, flatten(colors), gl.STATIC_DRAW);

    let vColor = gl.getAttribLocation(program, "vColor");
    gl.vertexAttribPointer(vColor, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vColor);

    let delta = gl.getUniformLocation(program, "delta");
    gl.uniform1f(delta, DELTA);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, vertices.length);

    shift += speed;

    if (count % 15 == 0) {
        count = 0;
        mouth_open = !mouth_open;
    }

    count += 1;

    if (shift > 110) shift = -33;
    else window.requestAnimationFrame(render);
};
