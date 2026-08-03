use warchest::board::board;
fn main() {
    let b = board();
    for i in 0..37 {
        println!("{:2} -> {:5}  loc={}", i, b.coord_str(i), b.is_location[i]);
    }
    println!("location_hexes: {:?}", b.location_hexes);
}
