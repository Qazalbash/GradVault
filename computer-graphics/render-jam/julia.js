"use strict";

let gl; // WebGL context
let clickX; // x coordinate of mouse click
let clickY; // y coordinate of mouse click
let vertices; // vertices of the quad
let mouseDown; // is the mouse down
let itterationDepth = 150; // max number of iterations
let viewProjectionMatrix; // view projection matrix

const projection = [1, 0, 0, 0, -1, 0, 0, 0, 1]; // projection matrix
const camera = {
    x: 0.0, // x position of camera
    y: 0.0, // y position of camera
    rotation: 0, // rotation of camera
    zoom: 1.0, // zoom of camera
};

window.onload = () => {
    let canvas = document.getElementById("gl-canvas");
    gl = canvas.getContext("webgl2");

    if (!gl) alert("WebGL 2.0 isn't available");

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);

    vertices = [
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, -1.0),
    ];

    render();

    const getClipSpaceMousePosition = (e) => {
        // get canvas relative css position
        const rect = canvas.getBoundingClientRect();
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;

        // get normalized 0 to 1 position across and down canvas
        const normalizedX = cssX / canvas.clientWidth;
        const normalizedY = cssY / canvas.clientHeight;

        // convert to bounding volume space
        const clipX = normalizedX * 2 - 1;
        const clipY = normalizedY * -2 + 1;

        return [clipX, clipY];
    };

    canvas.addEventListener("wheel", (e) => {
        const newZoom = camera.zoom * Math.pow(2, e.deltaY * -0.01);
        camera.zoom = Math.max(0.02, Math.min(100000000, newZoom));

        e.preventDefault();

        const clipCoordinates = getClipSpaceMousePosition(e);
        const [preZoomX, preZoomY] = transformPoint(
            flatten(inverse3(viewProjectionMatrix)),
            clipCoordinates
        );

        updateViewProjection();

        const [postZoomX, postZoomY] = transformPoint(
            flatten(inverse3(viewProjectionMatrix)),
            clipCoordinates
        );

        camera.x += preZoomX - postZoomX;
        camera.y += preZoomY - postZoomY;

        render();
    });

    canvas.addEventListener("mousedown", (e) => (mouseDown = true));

    canvas.addEventListener("mouseup", (e) => (mouseDown = false));

    canvas.addEventListener("mousemove", (e) => {
        if (mouseDown) {
            const rect = canvas.getBoundingClientRect();
            e.preventDefault();

            clickX = (2 * (e.clientX - rect.left)) / canvas.width - 1;
            clickY = (2 * (rect.top - e.clientY)) / canvas.height + 1;

            render();
        }
    });
};

/**
 *
 * @param {mat3} m
 * @param {vec2} v
 * @returns {vec2} transformed point
 */
const transformPoint = (m, v) => {
    const v0 = v[0],
        v1 = v[1],
        d = v0 * m[2] + v1 * m[5] + m[8];

    return [
        (v0 * m[0] + v1 * m[3] + m[6]) / d,
        (v0 * m[1] + v1 * m[4] + m[7]) / d,
    ];
};
/**
 * Update the view projection matrix
 */
const updateViewProjection = () => {
    const projectionMatrix = mat3(projection),
        cameraMatrix = makeCameraMatrix();
    let viewMatrix = inverse3(cameraMatrix);

    viewProjectionMatrix = mult(projectionMatrix, viewMatrix);
};

/**
 *
 * @returns {mat3} camera matrix
 */
const makeCameraMatrix = () => {
    const zoomScale = 1 / camera.zoom;
    let cameraMatrix = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);

    cameraMatrix = mult(cameraMatrix, translate(camera.x, camera.y));
    cameraMatrix = mult(
        cameraMatrix,
        rotate(camera.rotation, vec3(0, 0, 1), 3)
    );

    return mult(cameraMatrix, scale(zoomScale, zoomScale));
};

/**
 * Render the scene
 */
const render = () => {
    let program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STATIC_DRAW);

    let vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    let escapeMax = gl.getUniformLocation(program, "nt");
    gl.uniform1i(escapeMax, itterationDepth);

    let matrix = gl.getUniformLocation(program, "matrix");
    gl.uniformMatrix3fv(matrix, false, [1, 0, 0, 0, 1, 0, 0, 0, 1]);

    let c_x = gl.getUniformLocation(program, "c_x");
    gl.uniform1f(c_x, clickX);

    let c_y = gl.getUniformLocation(program, "c_y");
    gl.uniform1f(c_y, clickY);

    let c_zoom = gl.getUniformLocation(program, "c_zoom");
    gl.uniform1f(c_zoom, camera.zoom);

    let camera_x = gl.getUniformLocation(program, "camera_x");
    gl.uniform1f(camera_x, camera.x);

    let camera_y = gl.getUniformLocation(program, "camera_y");
    gl.uniform1f(camera_y, camera.y);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_FAN, 0, vertices.length);

    updateViewProjection();
};
