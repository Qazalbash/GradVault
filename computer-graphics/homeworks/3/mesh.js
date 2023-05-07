"use strick";

let vertices = [];
let faces = [];
let data = [];
let color_data = [];
let str_data = "";
let actual_verts = [];
let actual_data = [];
let actual_color_data = [];
let actual_faces = [];
let theta_x = 0;
let theta_y = 0;
let theta_z = 0;
let x_dist = 0;
let y_dist = 0;
let z_dist = 0;
let x_scl = 1;
let y_scl = 1;
let z_scl = 1;
let x_shear = Math.PI / 2;
let y_shear = Math.PI / 2;
let z_shear = Math.PI / 2;
document.getElementById('myfl').onchange = function() {
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    var myfl = this.files[0];
    var reader = new FileReader();
    reader.onload = function(progressEvent) {
        str_data = this.result.slice();
        extract_verts();

    };
    reader.readAsText(myfl);


};

function extract_verts() {
    vertices = [];
    color_data = [];
    data = [];
    faces = [];
    actual_verts = [];
    actual_data = [];
    actual_color_data = [];
    actual_faces = [];
    // for (var i=0; i< str_data.length; i++){
    //     console.log(str_data[i]);
    // }
    const result = str_data.split(/\r?\n/);
    for (var i = 0; i < result.length; i++) {
        result[i] = result[i].split(/\r? /);
    }
    var n;
    var f;
    var start = 0;
    var c = 0;
    var ins = 0;
    var ins_idx = 0;
    console.log(result);
    for (var i = 0; i < result.length; i++) {
        if (result[i].length == 3) {
            if (result[i][0] === "element" && result[i][1] == "vertex") {
                n = parseInt(result[i][2]);
            } else if (result[i][0] === "element" && result[i][1] == "face") {
                f = parseInt(result[i][2]);
            } else if (result[i][0] === "property") {
                c += 1;
                if (result[i][2] === "intensity") {
                    ins = 1;
                    ins_idx = c - 1;

                }
            }
        } else if (result[i][0] === "end_header") {
            start += i + 1;
        }
    }


    var max_y = 0;
    var max_x = 0;
    var max_z = 0;
    var min_y = 0;
    var min_x = 0;
    var min_z = 0;
    var max_i = 0;
    console.log(start);
    for (var j = start; j < n + start; j++) {
        var arr = [];
        for (var k = 0; k < c; k++) {
            arr.push(parseFloat(result[j][k]));
        }
        vertices.push(arr);
        if (arr[1] > max_y) {
            max_y = arr[1];
        }
        if (arr[0] > max_x) {
            max_x = arr[0];
        }
        if (arr[2] > max_z) {
            max_z = arr[2];
        }
        if (arr[1] < min_y) {
            min_y = arr[1];
        }
        if (arr[0] < min_x) {
            min_x = arr[0];
        }
        if (arr[2] < min_z) {
            min_z = arr[2];
        }
        if (arr[ins_idx] > max_i) {
            max_i = arr[ins_idx];
        }
    }
    var mult = 1 / max_i;
    var tx = 0 - (max_x + min_x) / 2;
    var ty = 0 - (max_y + min_y) / 2;
    var tz = 0 - (max_z + min_z) / 2;
    for (var i = 0; i < vertices.length; i++) {
        vertices[i][0] = tx + vertices[i][0];
        vertices[i][1] = ty + vertices[i][1];
        vertices[i][2] = tz + vertices[i][2];
        vertices[i][ins_idx] = mult * vertices[i][ins_idx];
    }

    max_y = 0;
    max_x = 0;
    max_z = 0;
    min_y = 0;
    min_x = 0;
    min_z = 0;


    for (var i = 0; i < vertices.length; i++) {
        if (vertices[i][1] > max_y) {
            max_y = vertices[i][1];
        }
        if (vertices[i][0] > max_x) {
            max_x = vertices[i][0];
        }
        if (vertices[i][2] > max_z) {
            max_z = vertices[i][2];
        }
        if (vertices[i][1] < min_y) {
            min_y = vertices[i][1];
        }
        if (vertices[i][0] < min_x) {
            min_x = vertices[i][0];
        }
        if (vertices[i][2] < min_z) {
            min_z = vertices[i][2];
        }
    }



    var rs;
    if (max_x > max_y) {
        rs = max_x;
    } else {
        rs = max_y;
    }
    var scl = 1 / rs;




    for (var j = start + n; j < f + n + start; j++) {
        var arr = [];
        for (var k = 1; k < 4; k++) {
            arr.push(parseFloat(result[j][k]));
        }
        faces.push(arr);
    }
    // console.log(faces);
    for (var i = 0; i < faces.length; i++) {
        var arr = [];
        for (var j = 0; j < faces[i].length; j++) {
            var a = vertices[faces[i][j]];
            data.push(vec3(scl * a[0], scl * a[1], scl * a[2]));
            var tmp = a[ins_idx] * 0.1;
            if (tmp > 1.0) {
                tmp = 1.0;
            }
            color_data.push(vec4(tmp, a[ins_idx] * 0.5, a[ins_idx] * 0.2, 1.0));

        }
    }
    // console.log(data);
    // console.log(mult);
    for (var i = 0; i < vertices.length; i++) {
        actual_verts.push([]);
        for (var j = 0; j < vertices[i].length; j++) {
            actual_verts[i].push(vertices[i][j])
        }
    }
    for (var i = 0; i < faces.length; i++) {
        actual_faces.push([]);
        for (var j = 0; j < faces[i].length; j++) {
            actual_faces[i].push(faces[i][j]);
        }
    }
    for (var i = 0; i < color_data.length; i++) {
        actual_color_data.push(vec4(color_data[i][0], color_data[i][1], color_data[i][2], color_data[i][3]));
        actual_data.push(vec3(data[i][0], data[i][1], data[i][2]));

    }
    render();
}

function rot_x() {
    theta_x += 0.1;
    render();




}

function rot_y() {
    theta_y += 0.1;

    render();
}

function rot_z() {
    theta_z += 0.1;

    render();
}

function trn_x() {
    x_dist += 0.1
        // for (var i = 0; i < data.length; i++) {
        //     data[i][0] = actual_data[i][0] + x_dist;
        // }
    render();

}

function trn_y() {
    y_dist += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = actual_data[i][1] + y_dist;
    // }
    render();
}

function trn_z() {
    z_dist += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = actual_data[i][2] + z_dist;
    // }
    render();
}

function trnb_x() {
    x_dist -= 0.1
        // for (var i = 0; i < data.length; i++) {
        //     data[i][0] = actual_data[i][0] + x_dist;
        // }
    render();

}

function trnb_y() {
    y_dist -= 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = actual_data[i][1] + y_dist;
    // }
    render();
}

function trnb_z() {
    z_dist -= 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = actual_data[i][2] + z_dist;
    // }
    render();
}

function scl_x() {
    x_scl += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][0] = x_scl * actual_data[i][0];
    // }
    render();
}

function scl_y() {
    y_scl += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = y_scl * actual_data[i][1];
    // }
    render();
}

function scl_z() {
    z_scl += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = z_scl * actual_data[i][2];
    // }
    render();
}

function scld_x() {
    x_scl -= 0.1;
    if (x_scl < 0) {
        x_scl = 0.0;
    }
    // for (var i = 0; i < data.length; i++) {
    //     data[i][0] = x_scl * actual_data[i][0];
    // }
    render();
}

function scld_y() {
    y_scl -= 0.1;
    if (y_scl < 0) {
        y_scl = 0.0;
    }
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = y_scl * actual_data[i][1];
    // }
    render();
}

function scld_z() {
    z_scl -= 0.1;
    if (z_scl < 0) {
        z_scl = 0.0;
    }
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = z_scl * actual_data[i][2];
    // }
    render();
}

function rfl_x() {
    x_scl = -1 * x_scl;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][0] = -1 * actual_data[i][0];
    // }
    render();
}

function rfl_y() {
    y_scl = -1 * y_scl;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = -1 * actual_data[i][1];
    // }
    render();
}

function rfl_z() {
    z_scl = -1 * z_scl;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = -1 * actual_data[i][2];
    // }
    render();
}

function shp_x() {
    x_shear -= 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][0] = actual_data[i][0] + actual_data[i][1] * (1 / Math.tan(x_shear));
    // }
    render();
}

function shp_y() {
    y_shear -= 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = actual_data[i][1] + actual_data[i][2] * (1 / Math.tan(y_shear));
    // }
    render();
}

function shp_z() {
    z_shear -= 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = actual_data[i][2] + actual_data[i][1] * (1 / Math.tan(z_shear));
    // }
    render();
}

function shn_x() {
    x_shear += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][0] = actual_data[i][0] + actual_data[i][1] * (1 / Math.tan(x_shear));
    // }
    render();
}

function shn_y() {
    y_shear += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][1] = actual_data[i][1] + actual_data[i][2] * (1 / Math.tan(y_shear));
    // }
    render();
}

function shn_z() {
    z_shear += 0.1;
    // for (var i = 0; i < data.length; i++) {
    //     data[i][2] = actual_data[i][2] + actual_data[i][1] * (1 / Math.tan(z_shear));
    // }
    render();
}

function reset() {
    data = [];
    color_data = [];
    faces = [];
    vertices = [];
    theta_x = 0;
    theta_y = 0;
    theta_z = 0;
    x_dist = 0;
    y_dist = 0;
    z_dist = 0;
    x_scl = 1;
    y_scl = 1;
    z_scl = 1;
    x_shear = Math.PI / 2;
    y_shear = Math.PI / 2;
    z_shear = Math.PI / 2;
    for (var i = 0; i < actual_verts.length; i++) {
        vertices.push([]);
        for (var j = 0; j < actual_verts[i].length; j++) {
            vertices[i].push(actual_verts[i][j])
        }
    }
    for (var i = 0; i < actual_faces.length; i++) {
        faces.push([]);
        for (var j = 0; j < actual_faces[i].length; j++) {
            faces[i].push(actual_faces[i][j]);
        }
    }
    for (var i = 0; i < actual_color_data.length; i++) {
        color_data.push(vec4(actual_color_data[i][0], actual_color_data[i][1], actual_color_data[i][2], actual_color_data[i][3]));
        data.push(vec3(actual_data[i][0], actual_data[i][1], actual_data[i][2]));

    }
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

    // almost_circle(Number(document.getElementById("slider").value));

    render();
};



function apply_transformations() {


    for (var i = 0; i < data.length; i++) {
        // Scaling
        data[i][0] = x_scl * actual_data[i][0];
        data[i][1] = y_scl * actual_data[i][1];
        data[i][2] = z_scl * actual_data[i][2];
        // Shearing
        data[i][0] = data[i][0] + data[i][1] * (1 / Math.tan(x_shear));
        data[i][1] = data[i][1] + data[i][2] * (1 / Math.tan(y_shear));
        data[i][2] = data[i][2] + data[i][1] * (1 / Math.tan(z_shear));
        // Rotating
        var tmpx = data[i][0];
        var tmpy = data[i][1];
        var tmpz = data[i][2];
        data[i][2] = tmpz * Math.cos(theta_x) - tmpy * Math.sin(theta_x);
        data[i][1] = tmpy * Math.cos(theta_x) + tmpz * Math.sin(theta_x);
        tmpx = data[i][0];
        tmpy = data[i][1];
        tmpz = data[i][2];
        data[i][0] = tmpx * Math.cos(theta_y) - tmpz * Math.sin(theta_y);
        data[i][2] = tmpz * Math.cos(theta_y) + tmpx * Math.sin(theta_y);
        tmpx = data[i][0];
        tmpy = data[i][1];
        tmpz = data[i][2];
        data[i][0] = tmpx * Math.cos(theta_z) - tmpy * Math.sin(theta_z);
        data[i][1] = tmpy * Math.cos(theta_z) + tmpx * Math.sin(theta_z);
        // Translating
        data[i][2] = data[i][2] + z_dist;
        data[i][1] = data[i][1] + y_dist;
        data[i][0] = data[i][0] + x_dist;
    }

}

const render = () => {
    apply_transformations();
    // console.log(vertices);
    let program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    // vertices
    let vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(data), gl.STATIC_DRAW);

    let vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    let cBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(color_data), gl.STATIC_DRAW);

    let cPosition = gl.getAttribLocation(program, "cPosition");
    gl.vertexAttribPointer(cPosition, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(cPosition);

    // rendering
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, data.length);
};