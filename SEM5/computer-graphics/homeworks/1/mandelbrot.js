"use strict";

window.onload = () => {
    calculate();
};

window.onchange = () => {
    calculate();
};

const abs = (x) => {
    return x < 0 ? -x : x;
};

const calculate = () => {
    const L = Number(document.getElementById("L").value);
    const x = Number(document.getElementById("x").value);
    const y = Number(document.getElementById("y").value);

    const r_web = map_point(0, L, -1, 1, x);
    const i_web = map_point(0, L, -1, 1, y);

    const r = 2 * r_web;
    const i = 2 * i_web;

    const sign = i >= 0 ? "+" : "-";

    document.getElementById("output-number").innerHTML = r + sign + abs(i) + "j";
    document.getElementById("output-webgl").innerHTML = "<" + r_web + "," + i_web + "," + 0.0 + "," + 1.0 + ">";
};
