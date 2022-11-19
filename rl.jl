
using IntervalSets
using ReinforcementLearning

mutable struct PodRacerEnv <: AbstractEnv
    state::Tuple{Float64,Tuple{FLoat64,Float64},Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}} # (angle, velocity, pos of next 2 checkpoints)
    reward::Float64
end

RLBase.action_space(env::PodRacerEnv) = ClosedInterval[0..100, -15..15]
RLBase.state_space(env::PodRacerEnv) = Space(ClosedInterval[0..100, -15..15])
RLBase.state(env::PodRacerEnv) = env.state
RLBase.reward(env::PodRacerEnv) = env.reward