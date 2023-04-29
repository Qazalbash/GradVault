"use strict";
var program;
var gl;
var to_rotate = 1;
var rotation = 0;
var divisions = 2;
var vertices = [];
var init_v = [];
var vertices_orig = [];
var colors = [];
var v = [
    vec3(0.0, 0.0, -1.0),
    vec3(0.0, 0.9428, 0.3333),
    vec3(-0.8165, -0.4714, 0.3333),
    vec3(0.8165, -0.4714, 0.3333),
];

var theta = 0;

function reset() {
    vertices = [...init_v];
    theta = 0;
}
function resume() {
    to_rotate = 1;
}

function pause() {
    to_rotate = 0;
}

function add_tetra_cooords(a, b, c, d) {
    var verts = [a, c, b, a, c, d, a, b, d, b, c, d];
    for (var i = 0; i < verts.length; i++) {
        vertices.push(verts[i]);
    }
    var cls = [
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 0.8),
        vec3(1.0, 0.0, 0.8),
    ];
    var idx = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    for (var i = 0; i < 12; i++) {
        colors.push(
            vec3(
                (cls[idx[i]][0] + Math.abs(verts[i][0])) / 2,
                cls[idx[i]][1] + Math.abs(verts[i][1]),
                cls[idx[i]][2] + Math.abs(verts[i][2]) / 2
            )
        );
    }
}

function tetrix(a, b, c, d, n) {
    if (n == 0) {
        add_tetra_cooords(a, b, c, d);
    } else {
        var m_ab = mix(a, b, 0.5);
        var m_bc = mix(b, c, 0.5);
        var m_ac = mix(a, c, 0.5);
        var m_ad = mix(a, d, 0.5);
        var m_bd = mix(b, d, 0.5);
        var m_cd = mix(c, d, 0.5);
        --n;
        tetrix(a, m_ab, m_ac, m_ad, n);
        tetrix(m_ab, b, m_bc, m_bd, n);
        tetrix(m_ac, m_bc, c, m_cd, n);
        tetrix(m_ad, m_bd, m_cd, d, n);
    }
}

window.onload = () => {
    let canvas = document.getElementById("gl-canvas");

    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL isn't available");

    tetrix(v[0], v[1], v[2], v[3], divisions);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl.DEPTH_TEST);
    vertices_orig = [...vertices];
    init_v = [...vertices];

    program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    render();
};

function transform_verts(theta) {
    for (var i = 0; i < vertices.length; i++) {
        if (rotation == 0) {
            vertices[i] = vec3(
                vertices[i][0],
                vertices_orig[i][1] * Math.cos(theta) -
                    vertices_orig[i][2] * Math.sin(theta),
                vertices_orig[i][2] * Math.cos(theta) +
                    vertices_orig[i][1] * Math.sin(theta)
            );
        } else if (rotation == 1) {
            vertices[i] = vec3(
                vertices_orig[i][0] * Math.cos(theta) -
                    vertices_orig[i][2] * Math.sin(theta),
                vertices[i][1],
                vertices_orig[i][2] * Math.cos(theta) +
                    vertices_orig[i][0] * Math.sin(theta)
            );
        } else if (rotation == 2) {
            vertices[i] = vec3(
                vertices_orig[i][0] * Math.cos(theta) -
                    vertices_orig[i][1] * Math.sin(theta),
                vertices_orig[i][1] * Math.cos(theta) +
                    vertices_orig[i][0] * Math.sin(theta),
                vertices[i][2]
            );
        }
    }
}

function set_depth(n) {
    console.log;
    divisions = n;
    vertices = [];
    tetrix(v[0], v[1], v[2], v[3], divisions);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    vertices_orig = [...vertices];
    init_v = [...vertices];
    theta = 0;
}

function rot_x() {
    vertices_orig = [...vertices];
    theta = 0;
    rotation = 0;
}

function rot_y() {
    vertices_orig = [...vertices];
    theta = 0;
    rotation = 1;
}

function rot_z() {
    vertices_orig = [...vertices];
    theta = 0;
    rotation = 2;
}

let render = () => {
    transform_verts(theta);

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

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, vertices.length);
    if (to_rotate == 1) theta += 0.05;

    if (theta > 2 * Math.PI) theta = 0;

    window.requestAnimationFrame(render);
};
