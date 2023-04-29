"use strick";

let vertices,
    points = [vec2(-1.0, -1.0), vec2(0.0, 1.0), vec2(1.0, -1.0)];

window.onload = () => {
    // fetching the html element
    let canvas = document.getElementById("gl-canvas");

    // getting the webgl2 context
    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL isn't available");

    // setting the viewing port and the default color of the canvas
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);

    render();
};

window.onchange = () => render();

const SierpinskisTriangle = (a, b, c, count) => {
    if (count === 0) vertices.push(a, b, c);
    else {
        // subdividing each side into half
        const ab = mix(a, b, 0.5);
        const ac = mix(a, c, 0.5);
        const bc = mix(b, c, 0.5);

        // decrementing the counter by 1
        count--;

        // calculating the coordinates for subtriangles
        SierpinskisTriangle(a, ab, ac, count);
        SierpinskisTriangle(ab, b, bc, count);
        SierpinskisTriangle(ac, bc, c, count);
    }
};

const render = () => {
    vertices = [];

    SierpinskisTriangle(
        points[0],
        points[1],
        points[2],
        Number(document.getElementById("slider").value)
    );

    let program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    // vertices
    let vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

    let vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    // rendering
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, vertices.length);
};
