using ReinforcementLearning

mutable struct PodRacerEnv <: AbstractEnv
    pos::Tuple{Float64,Float64}
    vel::Tuple{Float64,Float64}
    angle::Float64
    checkpoints::Vector{Tuple{Float64,Float64}}
    current_checkpoint::Int
    reward::Float64
    max_angle_change::Float64 = 15
    checkpoints_size::Float64 = 150
    time_step::Float64 = 0.1
end

RLBase.action_space(env::PodRacerEnv) = env.