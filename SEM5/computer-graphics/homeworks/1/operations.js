"use-strict";

var vect1, vect2, n, lst, zero_vec;

function valid_length(a) {
    for (var i = 0; i < a.length; i++)
        if (!(a[i] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
            return 0;
    if (Number(a) >= 2 && Number(a) <= 4) return 1;
    return 2;
}

function Take_Input() {
    while (true) {
        n = window.prompt("Input the Dimension of the vectors:");
        if (valid_length(n) == 0) alert("Invalid Vector length. Try again !");
        else if (valid_length(n) == 2)
            alert("Length not within the specified range. Try again !");
        else break;
    }

    lst = ["x", "y", "z", "w"];
    var v1 = [],
        v2 = [];
    for (var i = 0; i < n; i++) {
        var tmp;
        while (true) {
            tmp = window.prompt(
                "Input the " + lst[i] + " component of the First Vector:"
            );
            if (!Number(tmp)) alert("Invalid value. Try again");
            else break;
        }
        v1.push(Number(tmp));
    }
    for (var i = 0; i < n; i++) {
        var tmp;
        while (true) {
            tmp = window.prompt(
                "Input the " + lst[i] + " component of the Second Vector:"
            );
            if (!Number(tmp)) alert("Invalid value. Try again");
            else break;
        }
        v2.push(Number(tmp));
    }
    if (n == 2) {
        vect1 = vec2(v1);
        vect2 = vec2(v2);
        zero_vec = vec2(0.0, 0.0);
    } else if (n == 3) {
        vect1 = vec3(v1);
        vect2 = vec3(v2);
        zero_vec = vec3(0.0, 0.0, 0.0);
    } else {
        vect1 = vec4(v1);
        vect2 = vec4(v2);
        zero_vec = vec4(0.0, 0.0, 0.0, 0.0);
    }
}

window.onload = () => {
    Take_Input();
};

function display_vec(vt) {
    var str1 = "";
    if (n >= 2) str1 += "( " + vt[0].toFixed(2) + ", " + vt[1].toFixed(2);
    if (n >= 3) str1 += ", " + vt[2].toFixed(2);
    if (n == 4) str1 += ", " + vt[3].toFixed(2);
    str1 += " )";
    return str1;
}

function Execute_Option(a) {
    if (a == 1) {
        if (equal(vect1, vect2)) alert("Yes, the vectors are equal !");
        else alert("No, the vectors are not equal !");
    } else if (a == 2) alert("The length of the vectors are " + n);
    else if (a == 3) {
        var tmp1 = add(vect1, zero_vec);
        var tmp2 = add(vect2, zero_vec);
        var norm1 = normalize(tmp1);
        var norm2 = normalize(tmp2);
        var str1 = "Normalized Vector 1: ";
        var str2 = "Normalized Vector 2: ";
        str1 += display_vec(norm1);
        str2 += display_vec(norm2);
        alert(str1 + "\n" + str2);
    } else if (a == 4)
        alert("The sum of the vectors is " + display_vec(add(vect1, vect2)));
    else if (a == 5)
        alert(
            "The difference of the vectors is " +
                display_vec(subtract(vect1, vect2))
        );
    else if (a == 6)
        alert(
            "The dot product of the vectors is " + dot(vect1, vect2).toFixed(2)
        );
    else if (a == 7) {
        if (n == 3)
            alert(
                "The cross product of the vectors is " +
                    display_vec(cross(vect1, vect2))
            );
        else
            alert(
                "Cannot perform cross product on vectors of dimension other than 3"
            );
    } else if (a == 8) window.close();
}
