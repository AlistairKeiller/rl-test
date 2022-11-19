
using IntervalSets
using ReinforcementLearning

mutable struct PodRacerEnv <: AbstractEnv
    pos::Tuple{Float64,Float64}
    vel::Tuple{Float64,Float64}
    angle::Tuple{Float64,Float64}
    checkpoints::Vector{Tuple{Float64,Float64}}
    current_checkpoint::Int
    state::Tuple{Float64,FLoat64,Float64,Float64,Float64,Float64,Float64} # (angle, vel1, vel2, distance1 of current checkpoint, distance2 of current checkpoint, distance1 of next checkpoint, distance2 of next checkpoint)
    reward::Float64
    is_terminated::Bool
end

RLBase.action_space(env::PodRacerEnv) = ClosedInterval[-15..15, 0..100] # angle, thrust
RLBase.state_space(env::PodRacerEnv) = Space(ClosedInterval[0..360, 0..1000, 0..1000, -3000..3000, -3000..3000, -3000..3000, -3000..3000])
RLBase.state(env::PodRacerEnv) = env.state
RLBase.reward(env::PodRacerEnv) = env.reward
RLBase.is_terminated(env::PodRacerEnv) = env.is_terminated
function RLBase.reset!(env::PodRacerEnv)
    env.pos = (rand(0:3000), rand(0:3000))
    env.vel = (0, 0)
    env.angle = rand(0:360)
    env.checkpoints = [(rand(0:3000), rand(0:3000)), (rand(0:3000), rand(0:3000)), (rand(0:3000), rand(0:3000))]
    env.current_checkpoint = 1
    env.reward = 0
    env.is_terminated = false
end
function (env::PodRacerEnv)(action)
    env.angle += action[1]
    env.vel += (thrust * cos(angle), thrust * sin(angle))
    env.pos += env.vel
    if norm(env.pos - env.checkpoints[env.current_checkpoint]) < 100
        env.current_checkpoint += 1
        if env.current_checkpoint > length(env.checkpoints)
            env.current_checkpoint = 1
        end
    end
end