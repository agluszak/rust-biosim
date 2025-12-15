// System to handle closing of the brain visualization window
use bevy::window::WindowClosed;

fn handle_brain_vis_window_closed(
    mut window_closed_events: MessageReader<WindowClosed>,
    windows: Query<(Entity, &Window), With<BrainVisWindowMarker>>,
    mut settings: ResMut<Settings>,
) {
    for event in window_closed_events.read() {
        for (entity, _window) in windows.iter() {
            if event.window == entity {
                // When the brain vis window is closed, disable the visualization
                settings.show_brain_visualization = false;
                println!(
                    "Brain visualization window closed, simulation paused or visualization disabled."
                );
            }
        }
    }
}
mod brain_vis;
mod genome;
mod neural_network;
mod settings;
mod specimen;

use crate::brain_vis::{
    BrainVisData,
    SelectedSpecimenResource,
    cleanup_brain_vis_text_system, // Corrected to render_brain_visualization_system
    render_brain_visualization_system,
    select_specimen_system,
    toggle_brain_vis_system,
    update_brain_vis_data,
};
use crate::settings::{MEMORY_SIZE, Settings};
use crate::specimen::{
    Age,
    Alive,
    Birthplace,
    Brain,
    BrainInputs,
    BrainOutputs,
    DeathTurn,
    Direction,
    Food,
    Genome,
    Health,
    Hunger,
    Memory,
    NeuronValue,
    NeuronValueConvertible,
    OriginalColor,
    Oscillator,
    Position,
    PreviousPosition,
    SimulationEntity,
    Size,
    SpecimenBundle,
    SpeedMultiplier,
    food_consumption_system, // Added food_consumption_system
};
use bevy::DefaultPlugins;
use bevy::app::{App, FixedUpdate, Startup, Update};
use bevy::input::keyboard::KeyCode;
use bevy::prelude::*;
// MeshMaterial2d now in prelude
use bevy::camera::{OrthographicProjection, Projection};
use bevy::time::Fixed;
use bevy::window::PresentMode;
// Removed: use bevy_prototype_lyon::prelude::*;
use bevy_vector_shapes::prelude::*; // Added
use kiddo::SquaredEuclidean;
use kiddo::float::kdtree::KdTree;
use parry2d::na::{Point2, Rotation2, Vector2, distance};
use rand::random;
use rand::seq::IteratorRandom;
use std::time::Instant;

// Define the KdTree type we'll use
// Bucket size of 256 to handle large populations
#[derive(Resource)]
struct SpatialMap {
    food_tree: KdTree<f32, u64, 2, 256, u32>, // KdTree for food positions
    specimen_tree: KdTree<f32, u64, 2, 256, u32>, // KdTree for specimen positions
}

// Shared mesh handles for rendering
#[derive(Resource)]
struct SharedMeshes {
    circle: Handle<Mesh>,
}

impl Default for SpatialMap {
    fn default() -> Self {
        Self {
            food_tree: KdTree::new(),
            specimen_tree: KdTree::new(),
        }
    }
}

#[derive(Resource)]
struct LastUpdateTime(Instant);

impl Default for LastUpdateTime {
    fn default() -> Self {
        Self(Instant::now())
    }
}

fn should_update_simulation(settings: Res<Settings>, last_update: Res<LastUpdateTime>) -> bool {
    if !settings.slow_mode {
        return true;
    }
    let now = Instant::now();
    let elapsed = now.duration_since(last_update.0);
    elapsed.as_secs_f32() >= 0.3
}

fn update_timestamp(mut last_update: ResMut<LastUpdateTime>) {
    last_update.0 = Instant::now();
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Shape2dPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.9, 0.9, 0.9)))
        .insert_resource(Settings::default())
        .insert_resource(Turn(0))
        .insert_resource(Generation(0))
        .insert_resource(GenerationStartTime(Instant::now()))
        .insert_resource(SelectedSpecimenResource::default())
        .insert_resource(BrainVisData::default())
        .insert_resource(SpatialMap::default())
        .insert_resource(Time::<Fixed>::from_hz(60.0)) // 60 simulation ticks per second
        .add_systems(Startup, setup_app)
        // Simulation systems run at fixed timestep - ORDER MATTERS
        .add_systems(
            FixedUpdate,
            (
                time_system,                   // 1. Increment turn, age specimens
                update_spatial_map,            // 2. Rebuild KdTree for spatial queries
                food_detection_system_kdtree,  // 3. Detect food proximity for brain input
                brain_input_collection_system, // 4. Gather all sensor data for brains
                thinking_system,               // 5. Run neural networks
                doing_system,                  // 6. Execute brain outputs (set WantsToMove, etc.)
                movement_system,               // 7. Apply movement to entities with WantsToMove
                food_consumption_system,       // 8. Eat nearby food
                hunger_system,                 // 9. Decrease hunger, apply starvation damage
                damage_system,                 // 10. Apply age-related damage
                aging_system,                  // 11. Visual age effects (size/color)
                death_system,                  // 12. Kill specimens with 0 health
                mating_system,                 // 13. Reproduction for healthy specimens
                food_spawn_system,             // 14. Spawn new food
                corpse_despawn_system,         // 15. Remove old corpses
            ),
        )
        // Display/UI systems run every frame
        .add_systems(
            Update,
            (
                (transparency_system).run_if(rendering_enabled),
                display_system,
                select_specimen_system,
                toggle_brain_vis_system,
                update_brain_vis_data,
                render_brain_visualization_system,
                cleanup_brain_vis_text_system,
                draw_food_connections,
            ),
        )
        .add_systems(
            Update,
            (
                add_specimens_system,
                render_toggle_system,
                slow_mode_toggle_system,
                toggle_food_connections_system,
                restart_system,
                camera_control_system,
            ),
        )
        .run();
}

fn setup_app(
    mut commands: Commands,
    settings: Res<Settings>,
    materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // Spawn camera with initial zoom to see the world (100x100 units)
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: 0.5, // Start zoomed out to see more of the world
            ..OrthographicProjection::default_2d()
        }),
    ));

    // Create shared circle mesh for all specimens
    let circle_mesh = meshes.add(Circle::new(1.0)); // Unit circle, scaled per specimen
    commands.insert_resource(SharedMeshes {
        circle: circle_mesh.clone(),
    });

    first_generation_system(commands, settings, materials, circle_mesh);
}

// Camera control system - zoom with scroll, pan with middle mouse or arrow keys
fn camera_control_system(
    mut query: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut scroll_events: MessageReader<bevy::input::mouse::MouseWheel>,
    windows: Query<&Window>,
    time: Res<Time>,
) {
    let Ok((mut transform, mut projection)) = query.single_mut() else {
        return;
    };

    // Extract the orthographic projection scale
    let Projection::Orthographic(ortho) = projection.as_mut() else {
        return;
    };
    let scale = ortho.scale;

    // Zoom with mouse scroll
    for event in scroll_events.read() {
        let zoom_factor = 1.0 - event.y * 0.1;
        ortho.scale = (ortho.scale * zoom_factor).clamp(0.01, 5.0);
    }

    // Zoom with +/- keys
    if keyboard_input.pressed(KeyCode::Equal) || keyboard_input.pressed(KeyCode::NumpadAdd) {
        ortho.scale = (ortho.scale * 0.98).clamp(0.01, 5.0);
    }
    if keyboard_input.pressed(KeyCode::Minus) || keyboard_input.pressed(KeyCode::NumpadSubtract) {
        ortho.scale = (ortho.scale * 1.02).clamp(0.01, 5.0);
    }

    // Pan with arrow keys or WASD
    let pan_speed = 50.0 * scale * time.delta_secs();
    if keyboard_input.pressed(KeyCode::ArrowLeft)
        || keyboard_input.pressed(KeyCode::KeyA) && !keyboard_input.just_pressed(KeyCode::KeyA)
    {
        transform.translation.x -= pan_speed;
    }
    if keyboard_input.pressed(KeyCode::ArrowRight) || keyboard_input.pressed(KeyCode::KeyD) {
        transform.translation.x += pan_speed;
    }
    if keyboard_input.pressed(KeyCode::ArrowUp) || keyboard_input.pressed(KeyCode::KeyW) {
        transform.translation.y += pan_speed;
    }
    if keyboard_input.pressed(KeyCode::ArrowDown)
        || keyboard_input.pressed(KeyCode::KeyS) && !keyboard_input.just_pressed(KeyCode::KeyS)
    {
        transform.translation.y -= pan_speed;
    }

    // Pan with middle mouse drag
    if mouse_input.pressed(MouseButton::Middle)
        && let Ok(window) = windows.single()
        && let Some(_cursor) = window.cursor_position()
    {
        // Simple pan - move in direction of cursor offset from center
        let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
        if let Some(cursor) = window.cursor_position() {
            let offset = (cursor - center) * 0.001 * scale;
            transform.translation.x += offset.x;
            transform.translation.y -= offset.y; // Y is inverted in screen coords
        }
    }

    // Reset camera position with Home key
    if keyboard_input.just_pressed(KeyCode::Home) {
        transform.translation = Vec3::ZERO;
        ortho.scale = 0.5;
    }
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

fn slow_mode_toggle_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<Settings>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyS) {
        settings.slow_mode = !settings.slow_mode;
        println!(
            "Slow mode {}",
            if settings.slow_mode {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

fn toggle_food_connections_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<Settings>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyF) {
        settings.show_food_connections = !settings.show_food_connections;
        println!(
            "Food connections {}",
            if settings.show_food_connections {
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
struct GenerationStartTime(Instant);

#[derive(Component)]
struct TurnText;

#[inline]
fn map_range(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    assert!(
        from_min <= from_max,
        "from_min: {}, from_max: {}",
        from_min,
        from_max
    );
    assert!(to_min <= to_max, "to_min: {}, to_max: {}", to_min, to_max);
    assert!(
        value >= from_min && value <= from_max,
        "value: {}, from_min: {}, from_max: {}",
        value,
        from_min,
        from_max
    );
    let value = (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min;
    assert!(
        value >= to_min && value <= to_max,
        "value: {}, to_min: {}, to_max: {}",
        value,
        to_min,
        to_max
    );
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
            NeuronValue::from_linear(hunger.0, 0.0, 100.0),
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
    if turn.0.is_multiple_of(settings.turns_per_generation) {
        generation.0 += 1;
        println!(
            "[{}] Time: {}, alive: {}",
            generation.0,
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
    mut query: Query<
        (
            &Age,
            &mut Transform,
            &mut Size,
            &OriginalColor,
            &MeshMaterial2d<ColorMaterial>,
        ),
        With<Alive>,
    >,
    settings: Res<Settings>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (age, mut transform, mut size, original_color, material_handle) in query.iter_mut() {
        // Ensure age is within expected bounds
        let safe_age = age.0.min(settings.max_age);

        // Age affects size - specimens grow a bit with age, then shrink when very old
        let relative_age = safe_age as f32 / settings.max_age as f32;
        let size_factor = if relative_age < 0.3 {
            // Young specimens grow from 0.7 to 1.0
            map_range(relative_age, 0.0, 0.3, 0.7, 1.0)
        } else if relative_age > 0.7 {
            // Very old specimens shrink from 1.0 to 0.4
            // map_range requires to_min <= to_max, so we invert: map 0.7-1.0 to 0.4-1.0, then invert
            1.4 - map_range(relative_age, 0.7, 1.0, 0.4, 1.0)
        } else {
            // Middle-aged specimens maintain size
            1.0
        };

        // Update the size component with bounds checking
        size.0 = settings.specimen_size * size_factor;
        // Scale the mesh (unit circle scaled to desired size)
        transform.scale = Vec3::splat(size.0);

        // Optionally, change color with age (e.g., fade slightly)
        let mut new_color = original_color.0;
        if relative_age > 0.7 {
            // Older specimens might fade
            let fade_factor = map_range(relative_age, 0.7, 1.0, 1.0, 0.5);
            new_color = new_color.with_alpha(new_color.alpha() * fade_factor);
        }
        // Update material color
        if let Some(material) = materials.get_mut(&material_handle.0) {
            material.color = new_color;
        }
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
        position.0 += direction.0 * Vector2::new(effective_speed * settings.base_speed, 0.0);

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

fn text_update_system(turn: Res<Turn>, mut query: Query<&mut Text, With<TurnText>>) {
    for mut text in query.iter_mut() {
        text.0 = format!("Turn: {}", turn.0);
    }
}

fn display_system(
    _turn: Res<Turn>,
    mut query: Query<(&Position, &mut Transform), Without<Camera2d>>,
) {
    for (position, mut transform) in query.iter_mut() {
        transform.translation = Vec3::new(position.0.x, position.0.y, transform.translation.z);
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

fn first_generation_system(
    mut commands: Commands,
    settings: Res<Settings>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    circle_mesh: Handle<Mesh>,
) {
    info!("First generation");
    // Initialize the first generation with random genomes
    for _ in 0..settings.population {
        let genome = genome::Genome::random(settings.genome_length);
        let position = Position::random(&settings);
        commands.spawn(SpecimenBundle::new(
            Genome(genome),
            &settings.brain_inputs,
            &settings.brain_outputs,
            settings.internal_neurons,
            position,
            &mut materials,
            &settings,
            circle_mesh.clone(),
        ));
    }
}

fn damage_system(mut query: Query<(&mut Health, &Age), With<Alive>>, settings: Res<Settings>) {
    for (mut health, age) in query.iter_mut() {
        // Apply damage for old age starting at age 200 (previously 300)
        if age.0 > settings.old_age {
            health.0 -= (age.0 - settings.old_age) as f32 * settings.old_age_damage_rate;
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

fn transparency_system(
    query: Query<(&Health, &MeshMaterial2d<ColorMaterial>, Option<&Alive>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (health, material_handle, alive) in query.iter() {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            match alive {
                Some(_) => {
                    let health_value = health.0.clamp(0.0, 100.0);
                    let alpha = health_value / 100.0;
                    // Preserve original RGB, only change alpha
                    let original_rgb = material.color.to_srgba();
                    material.color = Color::srgba(
                        original_rgb.red,
                        original_rgb.green,
                        original_rgb.blue,
                        alpha,
                    );
                }
                None => {
                    // Dead, make it mostly transparent but slightly visible
                    let original_srgba = material.color.to_srgba();
                    material.color = Color::srgba(
                        original_srgba.red * 0.3, // Darken the color
                        original_srgba.green * 0.3,
                        original_srgba.blue * 0.3,
                        0.1, // Very low alpha
                    );
                }
            }
        }
    }
}

fn mating_system(
    mut commands: Commands,
    settings: Res<Settings>,
    alive: Query<(Entity, &Position, &Genome, &Age, &Health, &Hunger), With<Alive>>,
    _turn: Res<Turn>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    shared_meshes: Res<SharedMeshes>,
) {
    // Collect all living specimens
    let specimens: Vec<_> = alive.iter().collect();
    if specimens.is_empty() {
        return;
    }

    // Population cap to prevent KdTree overflow and maintain performance
    const MAX_POPULATION: usize = 2000;
    if specimens.len() >= MAX_POPULATION {
        return; // Don't reproduce if at capacity
    }

    // For each specimen, check if it's time to mate based on its age
    for (entity_a, pos_a, genome_a, age_a, health_a, hunger_a) in &specimens {
        // Check if the specimen has reached a mating age (every 150 turns)
        if age_a.0 % 150 == 0 && age_a.0 > 0 {
            // NATURAL SELECTION: Only healthy, well-fed specimens can reproduce
            // Require health > 60% and hunger > 40%
            if health_a.0 < 60.0 || hunger_a.0 < 40.0 {
                continue; // Too weak or hungry to reproduce
            }

            // Choose a random healthy mating partner
            if let Some((_entity_b, pos_b, genome_b, _, _, _)) = specimens
                .iter()
                .filter(|(e, _, _, _, health, hunger)| {
                    *e != *entity_a && health.0 >= 60.0 && hunger.0 >= 40.0
                })
                .choose(&mut rand::rng())
            {
                // Create new specimen near midpoint between parents with some randomness
                let midpoint_x = (pos_a.0.x + pos_b.0.x) / 2.0;
                let midpoint_y = (pos_a.0.y + pos_b.0.y) / 2.0;
                // Add random offset to prevent KdTree bucket overflow
                let offset_x: f32 = (random::<f32>() - 0.5) * 5.0;
                let offset_y: f32 = (random::<f32>() - 0.5) * 5.0;
                let new_x = (midpoint_x + offset_x)
                    .clamp(-settings.world_half_size, settings.world_half_size);
                let new_y = (midpoint_y + offset_y)
                    .clamp(-settings.world_half_size, settings.world_half_size);
                let new_position = Position(Point2::new(new_x, new_y));

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
                    &mut materials,
                    &settings,
                    shared_meshes.circle.clone(),
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
    _materials: ResMut<Assets<ColorMaterial>>, // Added
) {
    // Only spawn food at the specified interval
    if !turn.0.is_multiple_of(settings.food_spawn_interval) {
        return;
    }

    // Don't spawn more food if we've reached the maximum
    let current_food_count = food_query.iter().count();
    if current_food_count >= settings.max_food_entities {
        return;
    }

    // Spawn food at random position
    let food_position = Position::random(&settings);

    // Create food shape as a greenish square
    let food_size = 1.5; // Size in world units

    // Save position values before moving
    let food_x = food_position.0.x;
    let food_y = food_position.0.y;

    commands.spawn((
        Food,
        food_position,
        Sprite {
            color: Color::srgb(0.1, 0.7, 0.1), // Greenish food
            custom_size: Some(Vec2::splat(food_size)),
            ..default()
        },
        Transform::from_translation(Vec3::new(food_x, food_y, 0.0)),
        SimulationEntity,
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

// System to rebuild the KdTree every frame
fn update_spatial_map(
    mut spatial_map: ResMut<SpatialMap>,
    specimen_query: Query<(Entity, &Position), With<Alive>>,
    food_query: Query<(Entity, &Position), With<Food>>,
) {
    // Clear the existing trees
    spatial_map.specimen_tree = KdTree::new();
    spatial_map.food_tree = KdTree::new();

    // Add all specimens to the tree
    for (entity, position) in specimen_query.iter() {
        let pos = [position.0.x, position.0.y];
        spatial_map.specimen_tree.add(&pos, entity.to_bits());
    }

    // Add all food items to the tree
    for (entity, position) in food_query.iter() {
        let pos = [position.0.x, position.0.y];
        spatial_map.food_tree.add(&pos, entity.to_bits());
    }
}

// Updated food detection system that uses the KdTree for efficient lookup
fn food_detection_system_kdtree(
    mut specimen_query: Query<(Entity, &Position, &mut BrainInputs), With<Alive>>,
    spatial_map: Res<SpatialMap>,
    settings: Res<Settings>,
) {
    use neural_network::Input;

    for (_, position, mut brain_inputs) in specimen_query.iter_mut() {
        let specimen_pos = [position.0.x, position.0.y];
        let max_distance = settings.world_size * std::f32::consts::SQRT_2;

        // Find the closest food using the KdTree nearest_n
        let nearest = spatial_map
            .food_tree
            .nearest_n::<SquaredEuclidean>(&specimen_pos, 1);

        let closest_distance = if !nearest.is_empty() {
            // Convert squared distance to normal distance
            nearest[0].distance.sqrt()
        } else {
            max_distance
        };

        // Add food proximity as brain input
        brain_inputs.add(
            Input::FoodProximity,
            NeuronValue::from_linear(closest_distance, 0.0, max_distance),
        );
    }
}

// Updated food consumption system that uses the KdTree for efficient lookup
fn food_consumption_system_kdtree(
    mut commands: Commands,
    mut specimen_query: Query<(Entity, &Position, &Size, &mut Hunger), With<Alive>>,
    _food_query: Query<(Entity, &Position), With<Food>>,
    spatial_map: Res<SpatialMap>,
    settings: Res<Settings>,
) {
    // Keep track of which food items have been consumed
    let mut consumed_food = Vec::new();

    for (_, position, size, mut hunger) in specimen_query.iter_mut() {
        let specimen_pos = [position.0.x, position.0.y];

        // Find food within eating range using KdTree
        let in_range_food = spatial_map
            .food_tree
            .within_unsorted::<SquaredEuclidean>(&specimen_pos, size.0);

        // Only consume one food item per turn - the closest one
        if !in_range_food.is_empty() && !consumed_food.contains(&in_range_food[0].item) {
            // Record that this food has been consumed
            consumed_food.push(in_range_food[0].item);

            // Restore hunger
            hunger.0 = (hunger.0 + settings.food_restore_amount).min(100.0);
        }
    }

    // Despawn all consumed food items
    for food_bits in consumed_food {
        // Convert the bits back to an Entity and despawn it
        let entity = Entity::from_bits(food_bits);
        commands.entity(entity).despawn();
    }
}

fn restart_system(
    mut commands: Commands,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut turn: ResMut<Turn>,
    mut generation: ResMut<Generation>,
    mut generation_start: ResMut<GenerationStartTime>,
    settings: Res<Settings>,
    simulation_entities: Query<Entity, With<SimulationEntity>>,
    materials: ResMut<Assets<ColorMaterial>>,
    shared_meshes: Res<SharedMeshes>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        // Despawn all simulation entities
        for entity in simulation_entities.iter() {
            commands.entity(entity).despawn();
        }

        // Reset counters
        turn.0 = 0;
        generation.0 = 0;
        generation_start.0 = Instant::now();

        // Use first_generation_system to spawn new specimens
        first_generation_system(commands, settings, materials, shared_meshes.circle.clone());
    }
}

// Marker component for line gizmos
#[derive(Component)]
struct LineGizmo;

// System to draw lines between specimens and their nearest food
fn draw_food_connections(
    mut painter: ShapePainter,
    specimen_query: Query<&Position, With<Alive>>,
    food_query: Query<(Entity, &Position), With<Food>>,
    spatial_map: Res<SpatialMap>,
    settings: Res<Settings>,
) {
    // Removed: Despawn all existing line gizmos

    // Only draw connections if the setting is enabled
    if !settings.show_food_connections {
        return;
    }

    painter.set_translation(Vec3::ZERO);
    painter.thickness = 1.5;
    painter.color = Color::srgba(1.0, 0.5, 0.0, 0.5); // Corrected to srgba

    for specimen_position in specimen_query.iter() {
        let specimen_pos = Vec2::new(specimen_position.0.x, specimen_position.0.y);

        // Find the closest food using the KdTree
        let nearest = spatial_map
            .food_tree
            .nearest_n::<SquaredEuclidean>(&[specimen_position.0.x, specimen_position.0.y], 1);

        if !nearest.is_empty() {
            let food_entity_bits = nearest[0].item;

            // Find the matching food entity by comparing entity bits
            for (food_entity, food_pos_data) in food_query.iter() {
                if food_entity.to_bits() == food_entity_bits {
                    let food_pos = Vec2::new(food_pos_data.0.x, food_pos_data.0.y);
                    painter.line(specimen_pos.extend(0.0), food_pos.extend(0.0));
                    break;
                }
            }
        }
    }
}

// Add new specimens when 'A' is pressed
fn add_specimens_system(
    mut commands: Commands,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    settings: Res<Settings>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    shared_meshes: Res<SharedMeshes>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyA) {
        println!("Adding {} new specimens", settings.add_specimens_count);

        // Spawn the configured number of new specimens with random genomes
        for _ in 0..settings.add_specimens_count {
            let genome = genome::Genome::random(settings.genome_length);
            let position = Position::random(&settings);
            commands.spawn(SpecimenBundle::new(
                Genome(genome),
                &settings.brain_inputs,
                &settings.brain_outputs,
                settings.internal_neurons,
                position,
                &mut materials,
                &settings,
                shared_meshes.circle.clone(),
            ));
        }
    }
}

// System to update the visibility of the brain visualization window
fn update_brain_vis_window_visibility(
    mut windows: Query<&mut Window>,
    settings: Res<Settings>,
    selected: Res<SelectedSpecimenResource>,
) {
    // Get the second window (brain visualization window)
    if let Some(mut window) = windows.iter_mut().nth(1) {
        // Show window only if brain visualization is enabled and a specimen is selected
        window.visible = settings.show_brain_visualization && selected.entity.is_some();
    }
}

// Create a separate window for brain visualization
fn create_brain_vis_window(mut commands: Commands, settings: Res<Settings>) {
    // Create a new window for brain visualization
    let window_entity = commands
        .spawn((
            Window {
                title: "Brain Visualization".to_string(),
                resolution: (
                    settings.brain_vis_window_width as u32,
                    settings.brain_vis_window_height as u32,
                )
                    .into(),
                present_mode: PresentMode::Immediate,
                visible: false, // Initially hidden until a specimen is selected
                ..default()
            },
            BrainVisWindowMarker,
        ))
        .id();

    // Add a dedicated camera for the brain visualization window, targeting the new window
    use bevy::camera::RenderTarget;
    use bevy::prelude::{Camera, Camera2d};
    use bevy::window::WindowRef;
    commands.spawn((
        Camera2d,
        Camera {
            target: RenderTarget::Window(WindowRef::Entity(window_entity)),
            ..default()
        },
        BrainVisCamera,
    ));
}

// Marker component for the brain visualization window
#[derive(Component)]
struct BrainVisWindowMarker;

// Component to mark the brain visualization camera
#[derive(Component)]
struct BrainVisCamera;
