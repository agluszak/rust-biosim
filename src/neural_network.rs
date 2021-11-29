#[derive(Debug, Clone)]
pub struct NeuralNetwork {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inputs {
    PosX,
    PosY,
    Direction,
    MoveSuccessful,
    Oscillator1,
    Oscillator2,
    Oscillator3,
    Age,
    Proximity,
    LongProbeType,
    Random,
    DistanceTravelled,
    DistanceToBirthplace,
    LongProbeDistance,
    Speed,
    Memory1,
    Memory2,
    Memory3,
    // Energy,
    // Health,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outputs {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveRandom,
    TurnLeftSlowly, // (15 degrees)
    TurnRightSlowly,
    TurnLeft, // (90 degrees)
    TurnRight,
    TurnAround, // (180 degrees)
    TurnRandom,
    SetOscillator1Period,
    SetOscillator2Period,
    SetOscillator3Period,
    SetLongProbeDistance,
    SetSpeed,
    SetMemory1,
    SetMemory2,
    SetMemory3,
}
