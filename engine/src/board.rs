//! Board geometry. Axial hex coordinates "x,y" as used in the site data.
//! The playable 2-player board is the radius-3 hexagon centered on (3,3):
//! 37 hexes with |x-3|<=3, |y-3|<=3, |x+y-6|<=3.
//!
//! Hexes are indexed 0..37 in row-major (y then x) order of the valid coords.
//! Everything downstream keys off the hex index; coords are only for I/O.

pub const N_HEXES: usize = 37;
/// The 10 control locations.
pub const N_LOCATIONS: usize = 10;
pub const NONE: u8 = 255;

/// Neighbor offsets in axial coords (matches the site's neighbor definition).
const NEIGHBOR_OFFSETS: [(i8, i8); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

/// The 10 location (base) coords, in the order used by the site data.
/// Indices 0,1 = white starts; 2,3 = black starts; 4..10 = neutral bases.
pub const LOCATION_COORDS: [(i8, i8); N_LOCATIONS] = [
    (4, 0),
    (6, 1), // white starts
    (0, 5),
    (2, 6), // black starts
    (2, 1),
    (3, 2),
    (5, 3),
    (1, 3),
    (3, 4),
    (4, 5), // neutral
];

pub struct Board {
    /// coord for each hex index.
    pub coord: [(i8, i8); N_HEXES],
    /// up to 6 neighbor hex indices per hex; NONE fills unused slots.
    pub neighbors: [[u8; 6]; N_HEXES],
    /// axial distance between every pair of hexes.
    pub dist: [[u8; N_HEXES]; N_HEXES],
    /// true if the hex is one of the 10 control locations.
    pub is_location: [bool; N_HEXES],
    /// the 10 location hex indices (parallel to LOCATION_COORDS).
    pub location_hexes: [u8; N_LOCATIONS],
    /// For each ordered pair (a,b): the single hex strictly between them iff
    /// they are exactly 2 apart in a straight line, else NONE. Used by
    /// Crossbowman / Archer / Lancer straight-line reasoning.
    pub between: [[u8; N_HEXES]; N_HEXES],
    /// For each ordered (from, dir): the hex one step from `from` in that of
    /// the 6 axial directions, or NONE if off-board. Used for straight lines.
    pub step: [[u8; 6]; N_HEXES],
}

fn valid_coord(x: i8, y: i8) -> bool {
    (x - 3).abs() <= 3 && (y - 3).abs() <= 3 && (x + y - 6).abs() <= 3
}

fn axial_dist(a: (i8, i8), b: (i8, i8)) -> u8 {
    let dx = (a.0 - b.0) as i32;
    let dy = (a.1 - b.1) as i32;
    ((dx.abs() + dy.abs() + (dx + dy).abs()) / 2) as u8
}

impl Board {
    pub fn new() -> Board {
        // Enumerate valid coords row-major (y ascending, x ascending).
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
        let mut step = [[NONE; 6]; N_HEXES];
        for i in 0..N_HEXES {
            let (x, y) = coord[i];
            for (d, (ox, oy)) in NEIGHBOR_OFFSETS.iter().enumerate() {
                let n = index_of(x + ox, y + oy);
                neighbors[i][d] = n;
                step[i][d] = n;
            }
        }

        let mut dist = [[0u8; N_HEXES]; N_HEXES];
        for a in 0..N_HEXES {
            for b in 0..N_HEXES {
                dist[a][b] = axial_dist(coord[a], coord[b]);
            }
        }

        // between[a][b]: the hex strictly between a and b when they are exactly
        // two apart along one of the 6 straight directions.
        let mut between = [[NONE; N_HEXES]; N_HEXES];
        for a in 0..N_HEXES {
            for d in 0..6 {
                let mid = step[a][d];
                if mid == NONE {
                    continue;
                }
                let far = step[mid as usize][d];
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
            step,
        }
    }

    pub fn coord_str(&self, hex: usize) -> String {
        let (x, y) = self.coord[hex];
        format!("{},{}", x, y)
    }

    /// Parse "x,y" into a hex index, or None if off-board / malformed.
    pub fn hex_of_coord(&self, s: &str) -> Option<usize> {
        let mut it = s.split(',');
        let x: i8 = it.next()?.trim().parse().ok()?;
        let y: i8 = it.next()?.trim().parse().ok()?;
        if it.next().is_some() {
            return None;
        }
        for i in 0..N_HEXES {
            if self.coord[i] == (x, y) {
                return Some(i);
            }
        }
        None
    }
}

use std::sync::OnceLock;
static BOARD: OnceLock<Board> = OnceLock::new();

/// Shared immutable board geometry. Built once.
pub fn board() -> &'static Board {
    BOARD.get_or_init(Board::new)
}

/// The trunk's neighbour gather: `[N_HEXES * 6]`, hex-major, in the fixed
/// axial direction order of `NEIGHBOR_OFFSETS`.
///
/// A missing neighbour is written as `N_HEXES`, not `NONE`: both Rust and torch
/// gather from a feature map padded to `N_HEXES + 1` rows whose last row is
/// zero, so an edge hex sums zeros in the directions it has no neighbour in and
/// nothing needs a mask. Direction order is preserved, which is what lets a
/// stack of blocks express the straight-line relations the unit cards are full
/// of.
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
