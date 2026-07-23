
fn main() {
    let ids = rlx_kittentts::ipa_to_ids(std::env::args().nth(1).as_deref().unwrap_or("həˈloʊ"));
    println!("{}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
}
