pub struct Settings {
    pub specimen_size: f32,
    pub world_size: f32,
    pub world_half_size: f32,
    pub population: usize,
    pub mutation_chance: f32,
    pub turns_per_generation: usize,
    pub proximity_distance: f32,
    pub default_longprobe_distance: f32,
}

impl Default for Settings {
    fn default() -> Self {
        let mut settings = Settings {
            specimen_size: 1.0,
            world_size: 100.0,
            world_half_size: 0.0,
            population: 100,
            mutation_chance: 0.01,
            turns_per_generation: 1000,
            proximity_distance: 3.0,
            default_longprobe_distance: 10.0,
        };
        settings.world_half_size = settings.world_size / 2.0;
        settings
    }
}
