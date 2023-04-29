"use strict";

const black = vec4(0.0, 0.0, 0.0, 1.0),
    white = vec4(1.0, 1.0, 1.0, 1.0),
    red = vec4(1.0, 0.0, 0.0, 1.0),
    green = vec4(0.0, 1.0, 0.0, 1.0),
    blue = vec4(0.0, 0.0, 1.0, 1.0);

window.onload = () => {
    calculate();
};

window.onchange = () => {
    calculate();
};

const calculate = () => {
    const x = Number(document.getElementById("x-value").value),
        width = Number(document.getElementById("width").value);

    if (x > width) alert("Invalid width and x-value");
    else {
        let rgb_color;
        if (2 * x < width) rgb_color = map_point(0, width / 2, red, green, x);
        else rgb_color = map_point(width / 2, width, green, blue, x);

        const bw = map_point(0, width, black, white, x),
            x_value = map_point(0.0, width, -1.0, 1.0, x);

        document.getElementById("output-bw").innerHTML = "<" + bw + ">";
        document.getElementById("output-rgb").innerHTML = "<" + rgb_color + ">";
        document.getElementById("output-x").innerHTML = x_value;
    }
};
