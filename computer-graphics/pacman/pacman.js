/**
 * returns the points for pacman centered at (x, y)
 * @param {Number} x
 * @param {Number} y
 * @param {Boolean} mouth_open
 * @returns {Object}
 */
const PACMAN = (x, y, mouth_open) => {
    const pacman = { vertices: [], colors: [] };

    let level = [
        vec3(-5, -2, 0),

        vec3(-5, 1, 1),
        vec3(-5, 1, -1),

        vec3(-5, 4, 2),
        vec3(-5, 4, -2),

        vec3(-4, 6, 3),
        vec3(-4, 6, -3),

        vec3(-4, 6, 4),
        vec3(-4, 6, -4),

        vec3(-3, 5, 5),
        vec3(-3, 5, -5),

        vec3(-1, 3, 6),
        vec3(-1, 3, -6),
    ];

    if (mouth_open) {
        // points required to draw the close mouth
        level = level.concat([
            vec3(-1, 7, 0),

            vec3(2, 7, 1),
            vec3(2, 7, -1),

            vec3(2, 7, -2),
            vec3(2, 7, 2),
        ]);
    }

    for (var i = 0; i < level.length; ++i) {
        const lev = level[i];
        pacman.vertices = pacman.vertices.concat(
            strech(lev[0], lev[1], lev[2])
        );
    }

    // shifting the pacman
    pacman.vertices = pacman.vertices.map((v) => {
        return vec2(v[0] + x, v[1] + y);
    });

    // color the pacman
    pacman.colors = Array.from({ length: pacman.vertices.length }, (v, i) =>
        vec4(1.0, 1.0, 0.0, 1.0)
    );

    return pacman;
};
