"use strict";

let gl, points;
const nt = 100;

window.onload = () => {
	// fetching the html element
	let canvas = document.getElementById("gl-canvas");

	// getting the webgl2 context
	gl = canvas.getContext("webgl2");

	if (!gl) alert("WebGL isn't available");

	// setting the viewing port and the default color of the canvas
	gl.viewport(0, 0, canvas.width, canvas.height);
	gl.clearColor(1.0, 1.0, 1.0, 1.0);

	points = [vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)];

	render();
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

	// uniforms
	let u_maxIterations = gl.getUniformLocation(program, "nt");
	gl.uniform1i(u_maxIterations, nt);

	// rendering
	gl.clear(gl.COLOR_BUFFER_BIT);
	gl.drawArrays(gl.TRIANGLE_FAN, 0, points.length);
};
