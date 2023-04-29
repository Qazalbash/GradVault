"use strict";

var gl;
var c1 = 0.0;
var c2 = 0.0;

var vertices = [vec3(0.0, 1.0, 0.0), vec3(-1.0, -1.0, 0.0), vec3(1.0, -1.0, 0.0)];
var color1 = vec4(1.0, 0.0, 0.0, 1.0);
var color2 = vec4(0.0, 0.0, 1.0, 1.0);
const hexToRgb = (hex) => {
	let r = parseInt(hex.slice(1, 3), 16),
		g = parseInt(hex.slice(3, 5), 16),
		b = parseInt(hex.slice(5, 7), 16);
	return vec4(r / 255, g / 255, b / 255, 1.0);
};

function rnd_b() {
	color1 = vec4(Math.random() % 1, Math.random() % 1, Math.random() % 1, 1.0);
	render();
}

function rnd_i() {
	color2 = vec4(Math.random() % 1, Math.random() % 1, Math.random() % 1, 1.0);
	render();
}

function bc_set() {
	var b_c = document.getElementById("boundary-color");
	color1 = hexToRgb(b_c.value);
	render();
}

function ic_set() {
	var i_c = document.getElementById("interior-color");
	color2 = hexToRgb(i_c.value);
	render();
}

function tg_b() {
	color1 = vec4(1.0 - color1[0], 1.0 - color1[1], 1.0 - color1[2], 1.0);
	render();
}

function tg_i() {
	color2 = vec4(1.0 - color2[0], 1.0 - color2[1], 1.0 - color2[2], 1.0);
	render();
}

window.onload = () => {
	// fetching the html element
	let canvas = document.getElementById("gl-canvas");

	// getting the webgl2 context
	gl = canvas.getContext("webgl2");

	if (!gl) alert("WebGL isn't available");

	// setting the viewing port and the default color of the canvas
	gl.viewport(0, 0, canvas.width, canvas.height);
	gl.clearColor(0.0, 0.0, 0.0, 1.0);

	// width of the canvas
	const width = canvas.width;

	// rendering the lines
	render();
};

let render = () => {
	let program = initShaders(gl, "vertex-shader", "fragment-shader");
	gl.useProgram(program);

	// vertices
	let vBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

	let vPosition = gl.getAttribLocation(program, "vPosition");
	gl.vertexAttribPointer(vPosition, 3, gl.FLOAT, false, 0, 0);
	gl.enableVertexAttribArray(vPosition);

	var color1_l = gl.getUniformLocation(program, "bnd_col");
	gl.uniform4fv(color1_l, color1);
	var color2_l = gl.getUniformLocation(program, "int_col");
	gl.uniform4fv(color2_l, color2);

	gl.clear(gl.COLOR_BUFFER_BIT);
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertices.length);
};
