
using IntervalSets
using ReinforcementLearning
using Flux

mutable struct PodRacerEnv <: AbstractEnv
    pos::Tuple{Float64,Float64}
    vel::Tuple{Float64,Float64}
    angle::Float64
    checkpoints::Vector{Tuple{Float64,Float64}}
    current_checkpoint::Int
    reward::Float64
end

PodRacerEnv() = PodRacerEnv((0, 0), (0, 0), 0, [(0, 0)], 1, 0)

RLBase.action_space(env::PodRacerEnv) = ClosedInterval[-15..15, 0..100] # angle, thrust
RLBase.state_space(env::PodRacerEnv) = ClosedInterval[0..360, 0..1000, 0..1000, -3000..3000, -3000..3000, -3000..3000, -3000..3000] # (angle, vel1, vel2, distance1 of current checkpoint, distance2 of current checkpoint, distance1 of next checkpoint, distance2 of next checkpoint)
RLBase.state(env::PodRacerEnv) = (
    env.angle,
    env.vel...,
    (env.checkpoints[env.current_checkpoint] .- env.pos)...,
    (env.checkpoints[env.current_checkpoint%length(env.checkpoints)+1] .- env.pos)...
)
RLBase.reward(env::PodRacerEnv) = env.reward
RLBase.is_terminated(env::PodRacerEnv) = env.current_checkpoint > length(env.checkpoints)
function RLBase.reset!(env::PodRacerEnv)
    env.pos = (rand(0:3000), rand(0:3000))
    env.vel = (0, 0)
    env.angle = rand(0:360)
    env.checkpoints = [(rand(0:3000), rand(0:3000)), (rand(0:3000), rand(0:3000)), (rand(0:3000), rand(0:3000))]
    env.current_checkpoint = 1
    env.reward = 0
end
function (env::PodRacerEnv)(action)
    env.reward = 0
    if norm(env.pos - env.checkpoints[env.current_checkpoint]) < 100
        env.current_checkpoint += 1
        env.reward += 10
        if env.current_checkpoint > length(env.checkpoints)
            return
        end
    end

    env.angle += action[1]
    env.angle %= 360
    env.vel = env.vel .+ (action[2] * cos(angle), action[2] * sin(angle))
    env.vel /= 1.1
    env.reward += (norm(env.checkpoints[env.current_checkpoint] .- env.pos) - norm(env.checkpoints[env.current_checkpoint%length(env.checkpoints)+1] .- env.pos)) / 1000
    env.pos += env.vel
end


N_ENV = 8
UPDATE_FREQ = 32
env = MultiThreadEnv([
    PodRacerEnv() for i in 1:N_ENV
])
ns, na = length(state(env[1])), length(action_space(env[1]))
RLBase.reset!(env)
agent = Agent(
    policy=PPOPolicy(
        approximator=ActorCritic(
            actor=Chain(
                Dense(ns, 256, relu),
                Dense(256, na),
            ),
            critic=Chain(
                Dense(ns, 256, relu),
                Dense(256, 1),
            ),
            optimizer=ADAM(1e-3),
        ) |> gpu,
        γ=0.99f0,
        λ=0.95f0,
        clip_range=0.1f0,
        max_grad_norm=0.5f0,
        n_epochs=4,
        n_microbatches=4,
        actor_loss_weight=1.0f0,
        critic_loss_weight=0.5f0,
        entropy_loss_weight=0.001f0,
        update_freq=UPDATE_FREQ,
    ),
    trajectory=PPOTrajectory(;
        capacity=UPDATE_FREQ,
        state=Matrix{Float32} => (ns, N_ENV),
        action=Vector{Int} => (N_ENV,),
        action_log_prob=Vector{Float32} => (N_ENV,),
        reward=Vector{Float32} => (N_ENV,),
        terminal=Vector{Bool} => (N_ENV,)
    ),
)
ex = Experiment(agent, env, StopAfterStep(10_000), TotalBatchRewardPerEpisode(N_ENV), "Example pod racer env")
run(ex)