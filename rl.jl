
using IntervalSets
using ReinforcementLearning

mutable struct PodRacerEnv <: AbstractEnv
    state::Tuple{Float64,FLoat64,Float64,Float64,Float64,Float64,Float64} # (angle, vel1, vel2, pos1 of current checkpoint, pos2 of current checkpoint, pos1 of next checkpoint, pos2 of next checkpoint)
    reward::Float64
    is_terminated::Bool
end

RLBase.action_space(env::PodRacerEnv) = ClosedInterval[0..100, -15..15]
RLBase.state_space(env::PodRacerEnv) = Space(ClosedInterval[0..360, 0..200, 0..200, 0..3000, 0..3000, 0..3000, 0..3000])
RLBase.state(env::PodRacerEnv) = env.state
RLBase.reward(env::PodRacerEnv) = env.reward
RLBase.is_terminated(env::PodRacerEnv) = env.is_terminated
RLBase.reset!(env::PodRacerEnv) = env.reward = 0