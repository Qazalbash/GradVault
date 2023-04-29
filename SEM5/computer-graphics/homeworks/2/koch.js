"use strick";

let vertices;
const initial_points = {
    P: { x: 0, y: -0.5 },
    Q: { x: 0.5, y: 0.5 },
    R: { x: -0.5, y: 0.5 },
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

    render();
};

window.onchange = () => render(); // on every change of the slider, the render function is called

const koch = (A, B, count) => {
    let dx = B.x - A.x,
        dy = B.y - A.y,
        unit = Math.sqrt(dx * dx + dy * dy) / 3,
        t = Math.atan2(dy, dx),
        P = { x: A.x + dx / 3, y: A.y + dy / 3 },
        Q = { x: B.x - dx / 3, y: B.y - dy / 3 },
        R = {
            x: P.x + Math.cos(t - Math.PI / 3) * unit,
            y: P.y + Math.sin(t - Math.PI / 3) * unit,
        };

    if (count >= 0) {
        koch(A, P, count - 1);
        koch(P, R, count - 1);
        koch(R, Q, count - 1);
        koch(Q, B, count - 1);
    } else {
        vertices.push(
            vec2(A.x, A.y),
            vec2(P.x, P.y),
            vec2(R.x, R.y),
            vec2(Q.x, Q.y),
            vec2(B.x, B.y)
        );
    }
};

const render = () => {
    vertices = [];

    const n = Number(document.getElementById("slider").value) - 2;

    koch(initial_points.P, initial_points.Q, n);
    koch(initial_points.Q, initial_points.R, n);
    koch(initial_points.R, initial_points.P, n);

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
    gl.drawArrays(gl.LINE_LOOP, 0, vertices.length);
};
