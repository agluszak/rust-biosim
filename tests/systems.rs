use evolution::*;
use bevy::prelude::*;

#[test]
fn time_system_steps() {
    let mut app = App::new();
    app.insert_resource(Settings::default());
    app.init_resource::<Turn>();
    app.init_resource::<Generation>();
    app.init_resource::<GenerationStartTime>();
    app.world.spawn((Age(0), Alive));

    app.add_systems(Update, time_system);
    app.update();

    assert_eq!(app.world.resource::<Turn>().0, 1);
    let age = app.world.query::<&Age>().single(&app.world).0;
    assert_eq!(age, 1);
}

#[test]
fn hunger_system_steps() {
    let mut app = App::new();
    app.insert_resource(Settings::default());
    app.world.spawn((Hunger(1.0), Health(2.0), Alive));

    app.add_systems(Update, hunger_system);
    app.update();

    let (hunger, health) = app.world.query::<(&Hunger, &Health)>().single(&app.world);
    assert_eq!(hunger.0, 0.0);
    assert!(health.0 < 2.0);
}

#[test]
fn food_spawns_when_interval_matches() {
    let mut settings = Settings::default();
    settings.food_spawn_interval = 1;
    let mut app = App::new();
    app.insert_resource(settings);
    app.init_resource::<Turn>();

    app.add_systems(Update, food_spawn_system);
    app.update();

    let food_count = app.world.query::<&Food>().iter(&app.world).count();
    assert_eq!(food_count, 1);
}
