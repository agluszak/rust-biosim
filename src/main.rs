mod genome;
mod neural_network;
mod settings;
mod specimen;

use crate::settings::{MEMORY_SIZE, Settings};
use crate::specimen::{
    Age, Alive, Birthplace, Brain, BrainInputs, BrainOutputs, DeathTurn, Direction, 
    Food, Genome, Health, Hunger, Memory, NeuronValue, NeuronValueConvertible, Oscillator, 
    Position, PreviousPosition, SpecimenBundle, SpeedMultiplier, Size,
};
use bevy::DefaultPlugins;
use bevy::app::{App, Startup, Update};
use bevy::input::keyboard::KeyCode;
use bevy::prelude::*;
use bevy::window::PresentMode;
use bevy_prototype_lyon::prelude::*;
use parry2d::na::{Point2, Rotation2, Vector2, distance};
use rand::prelude::IndexedRandom;
use rand::random;
use rand::seq::IteratorRandom;
use std::time::Instant;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Turbo Evolution Giga Simulator".to_string(),
                resolution: (550., 550.).into(),
                present_mode: PresentMode::Immediate,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(ShapePlugin)
        .add_systems(Startup, (
            setup_system,
            first_generation_system.after(setup_system)
        ))
        .add_systems(Update, render_toggle_system)
        .add_systems(Update, display_system.run_if(rendering_enabled))
        .add_systems(Update, text_update_system.run_if(rendering_enabled))
        .add_systems(Update, transparency_system.run_if(rendering_enabled))
        .add_systems(Update, aging_system)
        .add_systems(Update, movement_system)
        .add_systems(Update, brain_input_collection_system)
        .add_systems(Update, thinking_system)
        .add_systems(Update, doing_system)
        .add_systems(Update, time_system)
        .add_systems(Update, food_spawn_system)
        .add_systems(Update, hunger_system)
        .add_systems(Update, food_detection_system)
        .add_systems(Update, food_consumption_system)
        .add_systems(
            Update,
            (
                damage_system,
                death_system,
                mating_system,
            )
                .chain(),
        )
        .add_systems(Update, corpse_despawn_system)
        .run();
}

// Check if rendering is enabled
fn rendering_enabled(settings: Res<Settings>) -> bool {
    settings.rendering_enabled
}

// Toggle rendering with the 'R' key
fn render_toggle_system(keyboard_input: Res<ButtonInput<KeyCode>>, mut settings: ResMut<Settings>) {
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        settings.rendering_enabled = !settings.rendering_enabled;
        println!(
            "Rendering {}",
            if settings.rendering_enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

#[derive(Component)]
struct WantsToMove;

#[derive(Resource)]
struct Turn(u32);

#[derive(Resource)]
struct Generation(u32);

#[derive(Resource)]
struct GenerationStartTime(std::time::Instant);

#[derive(Component)]
struct TurnText;

#[inline]
fn map_range(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    assert!(from_min <= from_max);
    assert!(to_min <= to_max);
    assert!(value >= from_min && value <= from_max);
    let value = (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min;
    assert!(value >= to_min && value <= to_max);
    value
}

fn brain_input_collection_system(
    mut query: Query<
        (
            &mut BrainInputs,
            &Position,
            &SpeedMultiplier,
            &Direction,
            &Birthplace,
            &Oscillator,
            &PreviousPosition,
            &Memory,
            &Age,
            &Hunger,
        ),
        With<Alive>,
    >,
    settings: Res<Settings>,
) {
    use neural_network::Input;
    for (
        mut brain_inputs,
        position,
        speed,
        direction,
        birthplace,
        _oscillator,
        previous_position,
        memory,
        age,
        hunger,
    ) in &mut query.iter_mut()
    {
        brain_inputs.add(
            Input::PosX,
            NeuronValue::from_linear(
                position.0.x,
                -settings.world_half_size,
                settings.world_half_size,
            ),
        );
        brain_inputs.add(
            Input::PosY,
            NeuronValue::from_linear(
                position.0.y,
                -settings.world_half_size,
                settings.world_half_size,
            ),
        );
        brain_inputs.add(Input::Speed, speed.get_neuron_value());
        brain_inputs.add(Input::DirectionX, direction.x());
        brain_inputs.add(Input::DirectionY, direction.y());
        brain_inputs.add(
            Input::DistanceToBirthplace,
            NeuronValue::from_linear(
                distance(&birthplace.0, &position.0),
                0.0,
                settings.world_size * std::f32::consts::SQRT_2,
            ),
        );
        brain_inputs.add(Input::Random, NeuronValue::from_linear(random(), 0.0, 1.0));
        brain_inputs.add(
            Input::DistanceTravelled,
            NeuronValue::from_linear(
                distance(&position.0, &previous_position.0).min(settings.base_speed * 2.0),
                0.0,
                settings.base_speed * 2.0,
            ),
        );

        // Ensure age is properly bounded
        let normalized_age = (age.0 as f32).min(settings.max_age as f32);
        brain_inputs.add(
            Input::Age,
            NeuronValue::from_linear(normalized_age, 0.0, settings.max_age as f32),
        );

        for i in 0..MEMORY_SIZE {
            // Clamp memory values to ensure they're in the valid range
            let memory_value = memory.0[i].clamp(-1.0, 1.0);
            brain_inputs.add(Input::Memory(i), NeuronValue::new(memory_value));
        }
        
        // Add hunger input
        brain_inputs.add(
            Input::Hunger,
            NeuronValue::from_linear(
                hunger.0,
                0.0,
                100.0,
            ),
        );
        
        // FoodProximity will be set in food_detection_system
    }
}

fn time_system(
    mut turn: ResMut<Turn>,
    mut generation: ResMut<Generation>,
    mut generation_start: ResMut<GenerationStartTime>,
    settings: Res<Settings>,
    mut query: Query<(&mut Age,), With<Alive>>,
) {
    if turn.0 % settings.turns_per_generation == 0 {
        generation.0 += 1;
        println!(
            "Time: {}, alive: {}",
            generation_start.0.elapsed().as_secs_f64(),
            query.iter().count()
        );
        generation_start.0 = Instant::now();
    }

    turn.0 += 1;
    for (mut age,) in query.iter_mut() {
        age.0 += 1;
    }
}

// Updated aging system - no longer modifies speed
fn aging_system(
    mut query: Query<(&Age, &mut Path, &mut Size), With<Alive>>,
    settings: Res<Settings>,
) {
    for (age, mut path, mut size) in query.iter_mut() {
        // Ensure age is within expected bounds
        let safe_age = age.0.min(settings.max_age);

        // Age affects size - specimens grow a bit with age, then shrink when very old
        let relative_age = safe_age as f32 / settings.max_age as f32;
        let size_factor = if relative_age < 0.3 {
            // Young specimens grow
            map_range(relative_age, 0.0, 0.3, 0.7, 1.0)
        } else if relative_age > 0.7 {
            // Very old specimens shrink
            map_range(relative_age, 0.7, 1.0, 1.0, 0.6)
        } else {
            // Middle-aged specimens maintain size
            1.0
        };

        // Update the size component with bounds checking
        size.0 = 10.0 * size_factor;

        // Update the visual representation
        let shape = shapes::Circle {
            radius: size.0,
            center: Vec2::new(0.0, 0.0),
        };
        *path = GeometryBuilder::build_as(&shape);
    }
}

fn thinking_system(mut query: Query<(&mut Brain, &BrainInputs, &mut BrainOutputs), With<Alive>>) {
    query
        .par_iter_mut()
        .for_each(|(mut brain, brain_inputs, mut brain_outputs)| {
            let outputs = brain.0.think(brain_inputs.read());
            *brain_outputs = BrainOutputs::from_hashmap(outputs);
        });
}

fn doing_system(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &BrainOutputs,
            &mut Direction,
            &mut SpeedMultiplier,
            &mut Memory,
        ),
        With<Alive>,
    >,
) {
    use neural_network::Output;
    for (entity, brain_outputs, mut direction, mut speed, mut memory) in &mut query.iter_mut() {
        if brain_outputs.activated(Output::Move) {
            commands.entity(entity).insert(WantsToMove);
        }
        if brain_outputs.activated(Output::Turn) {
            let desired_x = brain_outputs.get(Output::DesiredDirectionX);
            let desired_y = brain_outputs.get(Output::DesiredDirectionY);
            let desired_direction = Vector2::new(desired_x.value(), desired_y.value()).normalize();
            *direction = Direction(Rotation2::rotation_between(
                &Vector2::new(1.0, 0.0),
                &desired_direction,
            ));
        }
        if brain_outputs.activated(Output::ChangeSpeed) {
            let output = brain_outputs.get(Output::DesiredSpeed);
            speed.set_from_neuron_value(&output);
        }
        for i in 0..MEMORY_SIZE {
            if brain_outputs.activated(Output::Remember(i)) {
                let output = brain_outputs.get(Output::DesiredMemory(i));
                memory.0[i] = output.value();
            }
        }
    }
}

// Updated movement system with age-related speed calculation
fn movement_system(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &mut Position,
            &mut PreviousPosition,
            &SpeedMultiplier,
            &Direction,
            &Age,
        ),
        (With<WantsToMove>, With<Alive>),
    >,
    settings: Res<Settings>,
) {
    for (entity, mut position, mut previous_position, speed, direction, age) in query.iter_mut() {
        previous_position.0 = position.0;

        // Calculate age-related speed modifier
        let safe_age = age.0.min(settings.max_age);
        let age_factor = 1.0 - (safe_age as f32 / settings.max_age as f32).clamp(0.0, 0.9);

        // Apply age factor to speed without modifying the SpeedMultiplier component
        let effective_speed = speed.0 * age_factor;

        // Move with the age-adjusted speed
        position.0 +=
            direction.0 * parry2d::na::Vector2::new(effective_speed * settings.base_speed, 0.0);

        // Clamp position to world boundaries
        position.0.x = position
            .0
            .x
            .clamp(-settings.world_half_size, settings.world_half_size);
        position.0.y = position
            .0
            .y
            .clamp(-settings.world_half_size, settings.world_half_size);

        commands.entity(entity).remove::<WantsToMove>();
    }
}

fn text_update_system(
    turn: Res<Turn>,
    generation: Res<Generation>,
    mut query: Query<&mut Text, With<TurnText>>,
) {
    for mut text in query.iter_mut() {
        text.0 = format!("Generation: {}\nTurn: {}", generation.0, turn.0);
    }
}

const DISPLAY_SCALE: f32 = 5.0;

fn display_system(_turn: Res<Turn>, mut query: Query<(&Position, &mut Transform)>) {
    for (position, mut transform) in query.iter_mut() {
        // TODO scale
        transform.translation = Vec3::new(
            position.0.x * DISPLAY_SCALE,
            position.0.y * DISPLAY_SCALE,
            0.0,
        );
    }
}

fn setup_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    use bevy::prelude::*;
    commands.spawn(Camera2d);
    commands.insert_resource(Settings::default());
    commands.insert_resource(Turn(0));
    commands.insert_resource(Generation(0));
    commands.insert_resource(GenerationStartTime(Instant::now()));
    commands.spawn((
        Text::new(""),
        TextFont {
            font: asset_server.load("fonts/Roboto-Regular.ttf"),
            font_size: 100.0,
            ..default()
        },
        TextColor(Color::BLACK),
        Node {
            align_self: AlignSelf::FlexEnd,
            position_type: PositionType::Absolute,
            bottom: Val::Px(5.0),
            right: Val::Px(15.0),
            ..Default::default()
        },
        TurnText,
    ));
}

fn is_first_turn(generation: Res<Generation>, turn: Res<Turn>) -> bool {
    generation.0 == 0 && turn.0 == 0
}

fn first_generation_system(mut commands: Commands, settings: Res<Settings>) {
    info!("First generation");
    // Initialize the first generation with random genomes
    for _ in 0..settings.population {
        let genome = genome::Genome::random(settings.genome_length);
        let position = Position(parry2d::na::Point2::new(
            rand::random::<f32>() * settings.world_size - settings.world_half_size,
            rand::random::<f32>() * settings.world_size - settings.world_half_size,
        ));
        commands.spawn(SpecimenBundle::new(
            Genome(genome),
            &settings.brain_inputs,
            &settings.brain_outputs,
            settings.internal_neurons,
            position,
        ));
    }
}

fn damage_system(
    mut query: Query<(&Position, &mut Health, &Age), With<Alive>>,
    settings: Res<Settings>,
) {
    for (position, mut health, age) in query.iter_mut() {
        let damage = if position.0.x <= 0.0 || position.0.y <= 0.0 {
            let distance_from_goal =
                ((-position.0.x).max(0.0).powi(2) + (-position.0.y).max(0.0).powi(2)).sqrt();
            // Convert distance to damage - further means more damage
            4.0 * distance_from_goal / settings.world_half_size
        } else {
            0.0
        };
        health.0 -= damage;

        // Apply additional damage for old age
        if age.0 > 300 {
            health.0 -= (age.0 - 300) as f32 * settings.old_age_damage_rate;
        }

        // Apply death from old age if they've reached maximum age
        if age.0 >= settings.max_age {
            health.0 = 0.0;
        }
    }
}

fn death_system(
    mut commands: Commands,
    query: Query<(Entity, &Health), With<Alive>>,
    turn: Res<Turn>,
) {
    for (entity, health) in query.iter() {
        if health.0 <= 0.0 {
            commands
                .entity(entity)
                .remove::<Alive>()
                .insert(DeathTurn(turn.0)); // Record when the specimen died
        }
    }
}

fn transparency_system(mut query: Query<(&Health, &mut Fill, &mut Stroke, Option<&Alive>)>) {
    for (health, mut fill, mut stroke, alive) in query.iter_mut() {
        match alive {
            Some(_) => {
                let health_value = health.0.clamp(0.0, 100.0);
                let alpha = health_value / 100.0;
                let previous_color = fill.color.to_srgba();
                *fill = Fill::color(Color::srgba(
                    previous_color.red,
                    previous_color.green,
                    previous_color.blue,
                    alpha,
                ));
                *stroke = Stroke::new(Color::BLACK, 1.0);
            }
            None => {
                *fill = Fill::color(Color::NONE);
                *stroke = Stroke::new(Color::BLACK, 2.0);
            }
        }
    }
}

fn mating_system(
    mut commands: Commands,
    settings: Res<Settings>,
    alive: Query<(Entity, &Position, &Genome, &Age), With<Alive>>,
    _turn: Res<Turn>,
) {
    // Collect all living specimens
    let specimens: Vec<(Entity, &Position, &Genome, &Age)> = alive.iter().collect();
    if specimens.is_empty() {
        return;
    }

    // For each specimen, check if it's time to mate based on its age
    for (entity_a, pos_a, genome_a, age_a) in &specimens {
        // Check if the specimen has reached a mating age (every 150 turns)
        if age_a.0 % 150 == 0 && age_a.0 > 0 {
            // Choose a random mating partner from the living specimens
            if let Some((_entity_b, pos_b, genome_b, _)) = specimens
                .iter()
                .filter(|(e, _, _, _)| e != entity_a) // Don't mate with self
                .choose(&mut rand::thread_rng())
            {
                // Create new specimen at midpoint between parents
                let new_x = (pos_a.0.x + pos_b.0.x) / 2.0;
                let new_y = (pos_a.0.y + pos_b.0.y) / 2.0;
                let new_position = Position(parry2d::na::Point2::new(new_x, new_y));

                // Crossover and mutate the genomes
                let mut genome = genome::Genome::crossover(&genome_a.0, &genome_b.0);
                genome.mutate(settings.mutation_chance);

                // Spawn the new specimen
                commands.spawn(SpecimenBundle::new(
                    Genome(genome),
                    &settings.brain_inputs,
                    &settings.brain_outputs,
                    settings.internal_neurons,
                    new_position,
                ));
            }
        }
    }
}

// Add a system to despawn dead specimens after a configured delay
fn corpse_despawn_system(
    mut commands: Commands,
    query: Query<(Entity, &DeathTurn), Without<Alive>>,
    turn: Res<Turn>,
    settings: Res<Settings>,
) {
    for (entity, death_turn) in query.iter() {
        // Calculate how many turns have passed since the specimen died
        let turns_since_death = if turn.0 >= death_turn.0 {
            turn.0 - death_turn.0
        } else {
            // Handle case where turn counter reset during generation change
            turn.0 + settings.turns_per_generation - death_turn.0
        };

        // If the specified delay has passed, despawn the entity
        if turns_since_death >= settings.corpse_despawn_delay {
            commands.entity(entity).despawn();
        }
    }
}

// Spawn food at random locations
fn food_spawn_system(
    mut commands: Commands,
    turn: Res<Turn>,
    settings: Res<Settings>,
    food_query: Query<Entity, With<Food>>,
) {
    // Only spawn food at the specified interval
    if turn.0 % settings.food_spawn_interval != 0 {
        return;
    }
    
    // Don't spawn more food if we've reached the maximum
    let current_food_count = food_query.iter().count();
    if current_food_count >= settings.max_food_entities {
        return;
    }
    
    // Spawn food at random position
    let food_position = Position(parry2d::na::Point2::new(
        rand::random::<f32>() * settings.world_size - settings.world_half_size,
        rand::random::<f32>() * settings.world_size - settings.world_half_size,
    ));
    
    // Create food shape as a green rectangle
    let food_size = 5.0;
    let shape = shapes::Rectangle {
        extents: Vec2::new(food_size, food_size),
        ..default()
    };
    
    commands.spawn((
        Food,
        food_position,
        ShapeBundle {
            path: GeometryBuilder::build_as(&shape),
            ..default()
        },
        Fill::color(Color::srgb(0f32, 1f32, 0f32)),
        Stroke::new(Color::BLACK, 1.0),
    ));
}

// Handle hunger mechanics
fn hunger_system(
    mut query: Query<(&mut Hunger, &mut Health), With<Alive>>,
    settings: Res<Settings>,
) {
    for (mut hunger, mut health) in query.iter_mut() {
        // Decrease hunger over time
        hunger.0 = (hunger.0 - settings.hunger_decrease_rate).max(0.0);
        
        // Apply damage if starving
        if hunger.0 <= 0.0 {
            health.0 -= settings.hunger_damage_rate;
        }
    }
}

// Detect closest food for each specimen
fn food_detection_system(
    mut specimen_query: Query<(&Position, &mut BrainInputs), With<Alive>>,
    food_query: Query<&Position, With<Food>>,
    settings: Res<Settings>,
) {
    use neural_network::Input;
    
    for (specimen_pos, mut brain_inputs) in specimen_query.iter_mut() {
        // Find the closest food
        let mut closest_distance = f32::MAX;
        
        for food_pos in food_query.iter() {
            let distance = parry2d::na::distance(&specimen_pos.0, &food_pos.0);
            closest_distance = closest_distance.min(distance);
        }
        
        // If no food is found, set to maximum distance
        if closest_distance == f32::MAX {
            closest_distance = settings.world_size * std::f32::consts::SQRT_2;
        }
        
        // Add food proximity as brain input
        brain_inputs.add(
            Input::FoodProximity,
            NeuronValue::from_linear(
                closest_distance,
                0.0,
                settings.world_size * std::f32::consts::SQRT_2,
            ),
        );
    }
}

// Check for collisions between specimens and food
fn food_consumption_system(
    mut commands: Commands,
    mut specimen_query: Query<(&Position, &Size, &mut Hunger), With<Alive>>,
    food_query: Query<(Entity, &Position), With<Food>>,
    settings: Res<Settings>,
) {
    // Store all food items with a consumed flag
    let mut food_items: Vec<(Entity, Point2<f32>, bool)> = food_query
        .iter()
        .map(|(entity, pos)| (entity, pos.0, false))
        .collect();
    
    // For each specimen, find the closest unconsumed food item
    for (specimen_pos, specimen_size, mut hunger) in specimen_query.iter_mut() {
        let mut closest_food_idx = None;
        let mut closest_distance = f32::MAX;
        
        // Find closest unconsumed food
        for (idx, (_, food_pos, consumed)) in food_items.iter().enumerate() {
            if *consumed {
                continue; // Skip already consumed food
            }
            
            let distance = distance(&specimen_pos.0, food_pos);
            
            if distance < specimen_size.0 && distance < closest_distance {
                closest_distance = distance;
                closest_food_idx = Some(idx);
            }
        }
        
        // If found a food item within range, consume it
        if let Some(idx) = closest_food_idx {
            // Mark food as consumed
            food_items[idx].2 = true;
            
            // Restore hunger
            hunger.0 = (hunger.0 + settings.food_restore_amount).min(100.0);
        }
    }
    
    // Despawn all consumed food
    for (entity, _, consumed) in food_items {
        if consumed {
            commands.entity(entity).despawn();
        }
    }
}
