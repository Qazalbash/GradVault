"use strict";

let map_point = (P, Q, A, B, X) => {
    if (P == A && Q == B) {
        return mix(P, Q, X);
    }

    let alpha;

    if (typeof P == "number") {
        // if the first argument is a number, then subjecting it to the below formula
        alpha = (X - P) / (Q - P);
        if (typeof A == "number") {
            return A + alpha * (B - A);
        }
    } else {
        // if the first argument is not number, then using the x-coordinated for alpha
        alpha = (X - P[0]) / (Q[0] - P[0]);
    }
    // interpolating the point using the mix function
    return mix(A, B, alpha);
};
