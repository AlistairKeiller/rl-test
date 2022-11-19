
using IntervalSets
using ReinforcementLearning

mutable struct PodRacerEnv <: AbstractEnv
    state::Tuple{Float64,FLoat64,Float64,Float64,Float64,Float64,Float64} # (angle, vel1, vel2, distance1 of current checkpoint, distance2 of current checkpoint, distance1 of next checkpoint, distance2 of next checkpoint)
    pos::Tuple{Float64,Float64}
    vel::Tuple{Float64,Float64}
    angle::Tuple{Float64,Float64}
    checkpoints::Vector{Tuple{Float64,Float64}}
    current_checkpoint::Int
    reward::Float64
    is_terminated::Bool
end

RLBase.action_space(env::PodRacerEnv) = ClosedInterval[0..100, -15..15]
RLBase.state_space(env::PodRacerEnv) = Space(ClosedInterval[0..360, 0..200, 0..200, 0..3000, 0..3000, 0..3000, 0..3000])
RLBase.state(env::PodRacerEnv) = env.state
RLBase.reward(env::PodRacerEnv) = env.reward
RLBase.is_terminated(env::PodRacerEnv) = env.is_terminated
function RLBase.reset!(env::PodRacerEnv)
    env.checkpoints = [(rand(0 .. 3000), rand(0 .. 3000)) for i in rand(2:5)]
    env.state = (0, 0, 0, env.checkpoints[1][1], env.checkpoints[1][2], env.checkpoints[2][1], env.checkpoints[2][2])
    env.reward = 0
    env.is_terminated = false
    return env.state
end