"use strick";

let vertices = [];

window.onload = () => {
    // fetching the html element
    let canvas = document.getElementById("gl-canvas");

    // getting the webgl2 context
    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL isn't available");

    // setting the viewing port and the default color of the canvas
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);

    almost_circle(Number(document.getElementById("slider").value));

    render();
};

window.onchange = () => {
    vertices = [];

    almost_circle(Number(document.getElementById("slider").value));

    render();
};

const almost_circle = (n) => {
    /*
    We kept the radius to a quater because we want to draw a circle
    four circle in a row
    */
    const r = 1 / 4;

    for (var step = 0; step < n; step++) {
        // divinding the circle in 2^(n+2) parts
        const theta = (2 * Math.PI) / 2 ** (step + 2),
            // calculating the offset from the center
            dx = (2 * (step % 4) + 1) * r - 1,
            dy = 1 - r * (1 + 2 * Math.floor(step / 4));

        // calculating the vertices for each step
        for (var k = 0; k < 2 ** (step + 2); k++) {
            /*
            We are calculating two vertices at a time because we are
            using the LINES primitive and therefore to group the
            vertices in pairs we have to calculate two at a time
            */
            const x1 = r * Math.cos(theta * k) + dx,
                y1 = r * Math.sin(theta * k) + dy,
                x2 = r * Math.cos(theta * (k + 1)) + dx,
                y2 = r * Math.sin(theta * (k + 1)) + dy;

            // adding the vertices to the vertices array
            vertices.push(vec2(x1, y1), vec2(x2, y2));
        }
    }
};

const render = () => {
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
    gl.drawArrays(gl.LINES, 0, vertices.length);
};
