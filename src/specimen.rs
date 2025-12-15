use crate::{MEMORY_SIZE, SpatialMap, genome, map_range, neural_network};
use bevy::prelude::*;
// Mesh2d and MeshMaterial2d now in prelude
use crate::settings::Settings;
use rand::random;
use std::collections::HashMap;

#[derive(Component)]
pub struct Health(pub f32);

#[derive(Component, Debug)]
pub struct Hunger(pub f32);

#[derive(Component, Debug)]
pub struct SpeedMultiplier(pub f32);

impl SpeedMultiplier {
    const MAX: f32 = 2.0;
}

impl Default for SpeedMultiplier {
    fn default() -> Self {
        SpeedMultiplier(1.0)
    }
}

impl NeuronValueConvertible for SpeedMultiplier {
    fn set_from_neuron_value(&mut self, neuron_value: &NeuronValue) {
        self.0 = neuron_value.as_multiplier(Self::MAX);
    }

    fn get_neuron_value(&self) -> NeuronValue {
        NeuronValue::from_multiplier(self.0, Self::MAX)
    }
}

#[derive(Component, Debug)]
pub struct Size(pub f32);

#[derive(Component, Debug)]
pub struct Position(pub parry2d::na::Point2<f32>);

impl Position {
    pub fn random(settings: &crate::Settings) -> Self {
        Position(parry2d::na::Point2::new(
            random::<f32>() * settings.world_size - settings.world_half_size,
            random::<f32>() * settings.world_size - settings.world_half_size,
        ))
    }
}

#[derive(Component, Debug)]
pub struct Direction(pub parry2d::na::Rotation2<f32>);

impl Default for Direction {
    fn default() -> Self {
        Direction(parry2d::na::Rotation2::new(0.0))
    }
}

impl Direction {
    pub fn x(&self) -> NeuronValue {
        NeuronValue::new(self.0.angle().cos())
    }

    pub fn y(&self) -> NeuronValue {
        NeuronValue::new(self.0.angle().sin())
    }
}

#[derive(Component, Debug)]
pub struct PreviousPosition(pub parry2d::na::Point2<f32>);

#[derive(Component, Debug)]
pub struct Age(pub u32);

#[derive(Component, Debug)]
pub struct Birthplace(pub parry2d::na::Point2<f32>);

#[derive(Component)]
pub struct Memory(pub [f32; MEMORY_SIZE]);

#[derive(Component)]
pub struct Oscillator(pub [f32; 3]);

#[derive(Component)]
pub struct Genome(pub genome::Genome);

#[derive(Component)]
pub struct Brain(pub neural_network::NeuralNetwork);

#[derive(Component)]
pub struct Alive;

#[derive(Component)]
pub struct DeathTurn(pub u32); // Store the turn number when the specimen died

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NeuronInput(f32);

#[derive(Component, Default)]
pub struct BrainInputs(HashMap<neural_network::Input, f32>);

impl BrainInputs {
    pub fn add(&mut self, input: neural_network::Input, value: NeuronValue) {
        self.0.insert(input, value.value());
    }

    pub fn read(&self) -> &HashMap<neural_network::Input, f32> {
        &self.0
    }
}

pub trait NeuronValueConvertible {
    fn set_from_neuron_value(&mut self, neuron_value: &NeuronValue);
    fn get_neuron_value(&self) -> NeuronValue;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NeuronValue(f32);

impl NeuronValue {
    const MIN: f32 = -1.0;
    const MAX: f32 = 1.0;

    pub fn new(value: f32) -> Self {
        assert!((Self::MIN..=Self::MAX).contains(&value));
        NeuronValue(value)
    }

    pub fn value(&self) -> f32 {
        self.0
    }

    pub fn as_linear(&self, min: f32, max: f32) -> f32 {
        map_range(self.0, Self::MIN, Self::MAX, min, max)
    }

    pub fn from_linear(value: f32, min: f32, max: f32) -> Self {
        Self::new(map_range(value, min, max, Self::MIN, Self::MAX))
    }

    pub fn as_multiplier(&self, max: f32) -> f32 {
        max.powf(self.0)
    }

    pub fn from_multiplier(multiplier: f32, max: f32) -> Self {
        NeuronValue::new(multiplier.log(max))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn neuron_value_from_to_multiplier(neuron_value: f32, max: f32) {
        let from_value = NeuronValue::new(neuron_value);
        let multiplier = from_value.as_multiplier(max);
        let actual = NeuronValue::from_multiplier(multiplier, max);
        assert!((actual.value() - neuron_value).abs() < 0.00001);
    }

    #[test]
    fn test_neuron_value_from_to_multiplier() {
        neuron_value_from_to_multiplier(0.0, 2.0);
        neuron_value_from_to_multiplier(1.0, 2.0);
        neuron_value_from_to_multiplier(-1.0, 2.0);
        neuron_value_from_to_multiplier(0.33, 2.0);
        neuron_value_from_to_multiplier(0.5, 2.0);
        neuron_value_from_to_multiplier(0.66, 2.0);
        neuron_value_from_to_multiplier(-0.33, 2.0);
        neuron_value_from_to_multiplier(-0.5, 2.0);
        neuron_value_from_to_multiplier(-0.66, 2.0);
    }
}

#[derive(Component, Default)]
pub struct BrainOutputs(HashMap<neural_network::Output, f32>);

impl BrainOutputs {
    pub fn from_hashmap(outputs: HashMap<neural_network::Output, f32>) -> Self {
        BrainOutputs(outputs)
    }

    pub fn get(&self, output: neural_network::Output) -> NeuronValue {
        NeuronValue::new(self.0.get(&output).copied().unwrap_or(0.0))
    }

    pub fn activated(&self, output: neural_network::Output) -> bool {
        self.get(output).value() > 0.0 // TODO make this configurable
    }
}

#[derive(Bundle)]
pub struct SpecimenBundle {
    speed: SpeedMultiplier,
    position: Position,
    previous_position: PreviousPosition,
    direction: Direction,
    birthplace: Birthplace,
    age: Age,
    health: Health,
    hunger: Hunger,
    memory: Memory,
    oscillator: Oscillator,
    genome: Genome,
    brain: Brain,
    brain_inputs: BrainInputs,
    brain_outputs: BrainOutputs,
    size: Size,
    alive: Alive,
    simulation_entity: SimulationEntity,
    original_color: OriginalColor,
    // Visual components - circle mesh
    mesh: Mesh2d,
    material: MeshMaterial2d<ColorMaterial>,
    transform: Transform,
}

// Component to store the original color of a specimen
#[derive(Component)]
pub struct OriginalColor(pub Color);

impl SpecimenBundle {
    pub fn new(
        genome: Genome,
        brain_inputs_settings: &[neural_network::Input],
        brain_outputs_settings: &[neural_network::Output],
        internal_neurons: usize,
        position: Position,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        settings: &Res<Settings>,
        circle_mesh: Handle<Mesh>,
    ) -> Self {
        let brain = neural_network::NeuralNetwork::from_genome(
            &genome.0,
            brain_inputs_settings,
            brain_outputs_settings,
            internal_neurons,
        );
        let initial_previous_position = PreviousPosition(position.0);
        let initial_birthplace = Birthplace(position.0);
        let initial_size = settings.specimen_size;

        // Determine initial color based on genome
        let genes: Vec<_> = genome.0.genes().collect();
        let r = genes
            .first()
            .map_or(0.5, |g| g.weight() as f32 / i16::MAX as f32 * 0.5 + 0.5);
        let g = genes
            .get(1)
            .map_or(0.5, |g| g.weight() as f32 / i16::MAX as f32 * 0.5 + 0.5);
        let b = genes
            .get(2)
            .map_or(0.5, |g| g.weight() as f32 / i16::MAX as f32 * 0.5 + 0.5);
        let initial_color = Color::srgb(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0));

        // Create material for this specimen
        let material_handle = materials.add(ColorMaterial::from_color(initial_color));

        // Save position for transform before moving
        let pos_x = position.0.x;
        let pos_y = position.0.y;

        Self {
            speed: SpeedMultiplier::default(),
            position,
            previous_position: initial_previous_position,
            direction: Direction::default(),
            birthplace: initial_birthplace,
            age: Age(0),
            health: Health(100.0),
            hunger: Hunger(100.0),
            memory: Memory([0.0; MEMORY_SIZE]),
            oscillator: Oscillator([0.0; 3]),
            genome,
            brain: Brain(brain),
            brain_inputs: BrainInputs::default(),
            brain_outputs: BrainOutputs::default(),
            size: Size(initial_size),
            alive: Alive,
            simulation_entity: SimulationEntity,
            original_color: OriginalColor(initial_color),
            mesh: Mesh2d(circle_mesh),
            material: MeshMaterial2d(material_handle),
            transform: Transform::from_translation(Vec3::new(pos_x, pos_y, 0.0))
                .with_scale(Vec3::splat(initial_size)), // Scale the unit circle
        }
    }
}

// Function to spawn a specimen entity with all its components, including visual representation.
pub fn spawn_specimen(
    commands: &mut Commands,
    genome: Genome,
    brain_inputs_settings: &[neural_network::Input],
    brain_outputs_settings: &[neural_network::Output],
    internal_neurons: usize,
    position: Position,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    settings: &Res<Settings>,
    circle_mesh: Handle<Mesh>,
) {
    commands.spawn(SpecimenBundle::new(
        genome,
        brain_inputs_settings,
        brain_outputs_settings,
        internal_neurons,
        position,
        materials,
        settings,
        circle_mesh,
    ));
}

#[derive(Component)]
pub struct Food;

#[derive(Component)]
pub struct SimulationEntity; // Marker component for all simulation entities

// System to consume food and restore hunger
pub fn food_consumption_system(
    mut commands: Commands,
    mut specimen_query: Query<(Entity, &Position, &Size, &mut Hunger), With<Alive>>,
    food_query: Query<(Entity, &Position), With<Food>>,
    spatial_map: Res<SpatialMap>,
    settings: Res<Settings>,
) {
    let mut consumed_food_entities: Vec<Entity> = Vec::new();

    for (_specimen_entity, specimen_position, specimen_size, mut hunger) in
        specimen_query.iter_mut()
    {
        let specimen_pos_array = [specimen_position.0.x, specimen_position.0.y];

        // Find food within eating range using KdTree
        // The items in KdTree are u64 (Entity bits)
        let nearby_food_item_bits = spatial_map
            .food_tree
            .within_unsorted::<kiddo::SquaredEuclidean>(
                &specimen_pos_array,
                specimen_size.0 * specimen_size.0, // KdTree stores squared distances
            );

        if !nearby_food_item_bits.is_empty() {
            // For simplicity, consume the first food item found in range
            // A more complex logic could choose the closest or based on other criteria
            let food_to_consume_bits = nearby_food_item_bits[0].item;
            let food_entity_to_consume = Entity::from_bits(food_to_consume_bits);

            // Check if this food item has already been consumed in this tick by another specimen
            if !consumed_food_entities.contains(&food_entity_to_consume) {
                // Check if the entity actually exists and is food (optional, but good for safety)
                if food_query.get(food_entity_to_consume).is_ok() {
                    hunger.0 = (hunger.0 + settings.food_restore_amount).min(100.0);
                    consumed_food_entities.push(food_entity_to_consume);
                    // Despawn the food item immediately after one specimen consumes it
                    commands.entity(food_entity_to_consume).despawn();
                }
            }
        }
    }
    // Despawning is now handled inside the loop to prevent multiple consumptions in one tick.
}
