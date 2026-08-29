pub const N_UNITS: usize = 20;

pub const ARCHER: u8 = 0;
pub const BERSERKER: u8 = 1;
pub const CAVALRY: u8 = 2;
pub const CROSSBOWMAN: u8 = 3;
pub const ENSIGN: u8 = 4;
pub const FOOTMAN: u8 = 5;
pub const KNIGHT: u8 = 6;
pub const LANCER: u8 = 7;
pub const LIGHT_CAVALRY: u8 = 8;
pub const MARSHAL: u8 = 9;
pub const MERCENARY: u8 = 10;
pub const PIKEMAN: u8 = 11;
pub const SCOUT: u8 = 12;
pub const SWORDSMAN: u8 = 13;
pub const WARRIOR_PRIEST: u8 = 14;
pub const ROYAL_GUARD: u8 = 15;
pub const BERSERKER_V2: u8 = 16;
pub const FOOTMAN_V2: u8 = 17;
pub const WARRIOR_PRIEST_V2: u8 = 18;
pub const ROYAL_COIN: u8 = 19;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tactic {
    None,
    Archer,
    Cavalry,
    Crossbowman,
    Ensign,
    Footman,
    Lancer,
    LightCavalry,
    Marshal,
    RoyalGuard,
}

pub const N_TACTICS: usize = 10;

impl Tactic {
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Tactic::None => 0,
            Tactic::Archer => 1,
            Tactic::Cavalry => 2,
            Tactic::Crossbowman => 3,
            Tactic::Ensign => 4,
            Tactic::Footman => 5,
            Tactic::Lancer => 6,
            Tactic::LightCavalry => 7,
            Tactic::Marshal => 8,
            Tactic::RoyalGuard => 9,
        }
    }
}

pub const CARD_FEATS: usize = N_TACTICS + 12 + 2 + 1;

pub fn write_card_features(u: u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), CARD_FEATS);
    let d = def(u);
    out.fill(0.0);
    out[d.tactic.index()] = 1.0;
    let flags = [
        d.berserker_v1,
        d.berserker_v2,
        d.knight,
        d.mercenary,
        d.pikeman,
        d.royal_guard,
        d.scout,
        d.swordsman,
        d.warrior_priest,
        d.warrior_priest_v2,
        d.footman_v2,
        d.two_footmen,
    ];
    for (i, f) in flags.iter().enumerate() {
        out[N_TACTICS + i] = *f as u8 as f32;
    }
    out[N_TACTICS + 12] = d.no_normal_attack as u8 as f32;
    out[N_TACTICS + 13] = d.is_royal_coin as u8 as f32;
    out[N_TACTICS + 14] = d.coins as f32 / 5.0;
}

pub struct UnitDef {
    pub id: u16,
    pub name: &'static str,
    pub coins: u8,
    pub tactic: Tactic,
    pub berserker_v1: bool,
    pub berserker_v2: bool,
    pub knight: bool,
    pub mercenary: bool,
    pub pikeman: bool,
    pub royal_guard: bool,
    pub scout: bool,
    pub swordsman: bool,
    pub warrior_priest: bool,
    pub warrior_priest_v2: bool,
    pub footman_v2: bool,
    pub two_footmen: bool,
    pub no_normal_attack: bool,
    pub is_royal_coin: bool,
}

macro_rules! def {
    ($id:expr, $name:expr, $coins:expr, $tac:expr) => {
        UnitDef {
            id: $id,
            name: $name,
            coins: $coins,
            tactic: $tac,
            berserker_v1: false,
            berserker_v2: false,
            knight: false,
            mercenary: false,
            pikeman: false,
            royal_guard: false,
            scout: false,
            swordsman: false,
            warrior_priest: false,
            warrior_priest_v2: false,
            footman_v2: false,
            two_footmen: false,
            no_normal_attack: false,
            is_royal_coin: false,
        }
    };
}

pub static UNITS: [UnitDef; N_UNITS] = {
    let mut a = [
        def!(1, "Archer", 4, Tactic::Archer),
        def!(2, "Berserker", 5, Tactic::None),
        def!(3, "Cavalry", 4, Tactic::Cavalry),
        def!(4, "Crossbowman", 5, Tactic::Crossbowman),
        def!(5, "Ensign", 5, Tactic::Ensign),
        def!(6, "Footman", 5, Tactic::Footman),
        def!(7, "Knight", 4, Tactic::None),
        def!(8, "Lancer", 4, Tactic::Lancer),
        def!(9, "Light Cavalry", 5, Tactic::LightCavalry),
        def!(10, "Marshal", 5, Tactic::Marshal),
        def!(11, "Mercenary", 5, Tactic::None),
        def!(12, "Pikeman", 4, Tactic::None),
        def!(16, "Scout", 5, Tactic::None),
        def!(17, "Swordsman", 5, Tactic::None),
        def!(18, "Warrior Priest", 4, Tactic::None),
        def!(19, "Royal Guard", 5, Tactic::RoyalGuard),
        def!(52, "Berserker V2", 5, Tactic::None),
        def!(53, "Footman V2", 5, Tactic::Footman),
        def!(54, "Warrior Priest V2", 4, Tactic::None),
        def!(13, "Royal Coin", 1, Tactic::None),
    ];
    a[ARCHER as usize].no_normal_attack = true;
    a[BERSERKER as usize].berserker_v1 = true;
    a[FOOTMAN as usize].two_footmen = true;
    a[KNIGHT as usize].knight = true;
    a[LANCER as usize].no_normal_attack = true;
    a[MERCENARY as usize].mercenary = true;
    a[PIKEMAN as usize].pikeman = true;
    a[ROYAL_GUARD as usize].royal_guard = true;
    a[SCOUT as usize].scout = true;
    a[SWORDSMAN as usize].swordsman = true;
    a[WARRIOR_PRIEST as usize].warrior_priest = true;
    a[BERSERKER_V2 as usize].berserker_v2 = true;
    a[FOOTMAN_V2 as usize].two_footmen = true;
    a[FOOTMAN_V2 as usize].footman_v2 = true;
    a[WARRIOR_PRIEST_V2 as usize].warrior_priest = true;
    a[WARRIOR_PRIEST_V2 as usize].warrior_priest_v2 = true;
    a[ROYAL_COIN as usize].is_royal_coin = true;
    a
};

pub fn index_of_id(id: u16) -> Option<u8> {
    if id == 13 || id == 14 {
        return Some(ROYAL_COIN);
    }
    for (i, u) in UNITS.iter().enumerate() {
        if u.id == id {
            return Some(i as u8);
        }
    }
    None
}

#[inline]
pub fn def(idx: u8) -> &'static UnitDef {
    &UNITS[idx as usize]
}
