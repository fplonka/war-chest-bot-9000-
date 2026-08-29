pub const N_HEXES: usize = 37;
pub const N_LOCATIONS: usize = 10;
pub const NONE: u8 = 255;

const NEIGHBOR_OFFSETS: [(i8, i8); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

pub const LOCATION_COORDS: [(i8, i8); N_LOCATIONS] = [
    (4, 0),
    (6, 1),
    (0, 5),
    (2, 6),
    (2, 1),
    (3, 2),
    (5, 3),
    (1, 3),
    (3, 4),
    (4, 5),
];

pub struct Board {
    pub coord: [(i8, i8); N_HEXES],
    pub neighbors: [[u8; 6]; N_HEXES],
    pub dist: [[u8; N_HEXES]; N_HEXES],
    pub is_location: [bool; N_HEXES],
    pub location_hexes: [u8; N_LOCATIONS],
    pub between: [[u8; N_HEXES]; N_HEXES],
}

fn valid_coord(x: i8, y: i8) -> bool {
    (x - 3).abs() <= 3 && (y - 3).abs() <= 3 && (x + y - 6).abs() <= 3
}

fn axial_dist(a: (i8, i8), b: (i8, i8)) -> u8 {
    let dx = (a.0 - b.0) as i32;
    let dy = (a.1 - b.1) as i32;
    ((dx.abs() + dy.abs() + (dx + dy).abs()) / 2) as u8
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Board {
        let mut coords: Vec<(i8, i8)> = Vec::with_capacity(N_HEXES);
        for y in 0..=6i8 {
            for x in 0..=6i8 {
                if valid_coord(x, y) {
                    coords.push((x, y));
                }
            }
        }
        assert_eq!(coords.len(), N_HEXES, "board must have exactly 37 hexes");

        let mut coord = [(0i8, 0i8); N_HEXES];
        for (i, c) in coords.iter().enumerate() {
            coord[i] = *c;
        }

        let index_of = |x: i8, y: i8| -> u8 {
            for (i, c) in coord.iter().enumerate() {
                if c.0 == x && c.1 == y {
                    return i as u8;
                }
            }
            NONE
        };

        let mut neighbors = [[NONE; 6]; N_HEXES];
        for i in 0..N_HEXES {
            let (x, y) = coord[i];
            for (d, (ox, oy)) in NEIGHBOR_OFFSETS.iter().enumerate() {
                neighbors[i][d] = index_of(x + ox, y + oy);
            }
        }

        let mut dist = [[0u8; N_HEXES]; N_HEXES];
        for a in 0..N_HEXES {
            for b in 0..N_HEXES {
                dist[a][b] = axial_dist(coord[a], coord[b]);
            }
        }

        let mut between = [[NONE; N_HEXES]; N_HEXES];
        for a in 0..N_HEXES {
            for d in 0..6 {
                let mid = neighbors[a][d];
                if mid == NONE {
                    continue;
                }
                let far = neighbors[mid as usize][d];
                if far == NONE {
                    continue;
                }
                between[a][far as usize] = mid;
            }
        }

        let mut is_location = [false; N_HEXES];
        let mut location_hexes = [NONE; N_LOCATIONS];
        for (li, (x, y)) in LOCATION_COORDS.iter().enumerate() {
            let h = index_of(*x, *y);
            assert!(h != NONE, "location coord must be on board");
            is_location[h as usize] = true;
            location_hexes[li] = h;
        }

        Board {
            coord,
            neighbors,
            dist,
            is_location,
            location_hexes,
            between,
        }
    }
}

use std::sync::OnceLock;
static BOARD: OnceLock<Board> = OnceLock::new();

pub fn board() -> &'static Board {
    BOARD.get_or_init(Board::new)
}

pub fn neighbour_gather() -> Vec<u8> {
    let bd = board();
    let mut out = Vec::with_capacity(N_HEXES * 6);
    for h in 0..N_HEXES {
        for d in 0..6 {
            let n = bd.neighbors[h][d];
            out.push(if n == NONE { N_HEXES as u8 } else { n });
        }
    }
    out
}
