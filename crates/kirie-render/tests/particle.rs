//! Unit tests for the CPU particle simulation: each operator and initializer's
//! math on synthetic particles (deterministic steps), emitter rate/burst
//! timing, lifetime culling, and pool bounds (docs/render-architecture.md §7.3).

use kirie_render::particle::{
    Initial, Initializer, Operator, Particle, ParticleSim, Rng, SimConfig, SpawnCtx, StepCtx,
};
use kirie_scene::particle::{InstanceOverride, NamedStage, ParticleSystem};
use kirie_scene::value::Vec3;
use serde_json::json;

// ---- helpers ---------------------------------------------------------------

fn stage(value: serde_json::Value) -> NamedStage {
    NamedStage::parse(value.as_object().expect("object"))
}

fn synth() -> Particle {
    Particle {
        position: [0.0, 0.0, 0.0],
        velocity: [0.0, 0.0, 0.0],
        acceleration: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0],
        angular_velocity: [0.0, 0.0, 0.0],
        color: [1.0, 1.0, 1.0],
        alpha: 1.0,
        size: 10.0,
        lifetime: 2.0,
        age: 0.0,
        frame: 0.0,
        initial: Initial {
            color: [1.0, 1.0, 1.0],
            alpha: 1.0,
            size: 10.0,
        },
        seed: 0xABCD_1234,
    }
}

fn step_ctx<'a>(
    dt: f32,
    time: f32,
    cps: &'a [Vec3],
    overrides: &'a kirie_render::particle::Overrides,
) -> StepCtx<'a> {
    StepCtx {
        dt,
        time,
        overrides,
        control_points: cps,
    }
}

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
}

fn approx_v(a: Vec3, b: Vec3) {
    for i in 0..3 {
        approx(a[i], b[i]);
    }
}

fn ovr() -> kirie_render::particle::Overrides {
    kirie_render::particle::Overrides::default()
}

// ---- operators -------------------------------------------------------------

#[test]
fn movement_integrates_position_then_velocity() {
    // gravity "0 10 0" is Y-flipped to (0,-10,0) on read (docs §7.3).
    let op = Operator::compile(
        &stage(json!({"name":"movement","gravity":"0 10 0"})),
        1,
        &mut Rng::new(0),
    );
    let mut p = synth();
    p.velocity = [10.0, 0.0, 0.0];
    let o = ovr();
    op.apply(&mut p, &step_ctx(0.1, 0.0, &[[0.0; 3]], &o));
    // pos += vel*dt (using pre-gravity velocity)
    approx_v(p.position, [1.0, 0.0, 0.0]);
    // vel += gravity*dt*speed
    approx_v(p.velocity, [10.0, -1.0, 0.0]);
}

#[test]
fn movement_drag_scales_velocity() {
    let op = Operator::compile(&stage(json!({"name":"movement","drag":5.0})), 1, &mut Rng::new(0));
    let mut p = synth();
    p.velocity = [10.0, 0.0, 0.0];
    let o = ovr();
    op.apply(&mut p, &step_ctx(0.1, 0.0, &[[0.0; 3]], &o));
    // vel *= max(0, 1 - drag*dt) = 0.5
    approx_v(p.velocity, [5.0, 0.0, 0.0]);
    approx_v(p.position, [1.0, 0.0, 0.0]);
}

#[test]
fn speed_override_scales_gravity() {
    let op = Operator::compile(
        &stage(json!({"name":"movement","gravity":"0 10 0"})),
        1,
        &mut Rng::new(0),
    );
    let mut p = synth();
    let mut o = ovr();
    o.speed = 2.0;
    op.apply(&mut p, &step_ctx(0.1, 0.0, &[[0.0; 3]], &o));
    approx_v(p.velocity, [0.0, -2.0, 0.0]); // -10 * 0.1 * 2
}

#[test]
fn angularmovement_wraps_rotation() {
    let op = Operator::compile(
        &stage(json!({"name":"angularmovement","force":"0 0 100"})),
        1,
        &mut Rng::new(0),
    );
    let mut p = synth();
    p.angular_velocity = [0.0, 0.0, 100.0];
    let o = ovr();
    op.apply(&mut p, &step_ctx(1.0, 0.0, &[[0.0; 3]], &o));
    // rotation += 100 rad then wrapped to (-pi, pi]
    assert!(p.rotation[2] > -std::f32::consts::PI && p.rotation[2] <= std::f32::consts::PI);
}

#[test]
fn alphafade_fades_in_and_out() {
    let op = Operator::compile(
        &stage(json!({"name":"alphafade","fadeintime":0.5,"fadeouttime":0.5})),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth(); // lifetime 2.0
    p.age = 0.5; // lp = 0.25 -> factor 0.5
    op.apply(&mut p, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
    approx(p.alpha, 0.5);
    let mut q = synth();
    q.age = 1.0; // lp 0.5 -> full
    op.apply(&mut q, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
    approx(q.alpha, 1.0);
    let mut r = synth();
    r.age = 1.5; // lp 0.75 -> factor 0.5
    op.apply(&mut r, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
    approx(r.alpha, 0.5);
}

#[test]
fn sizechange_ramps_linearly() {
    let op = Operator::compile(
        &stage(json!({"name":"sizechange","starttime":0.0,"endtime":1.0,"startvalue":1.0,"endvalue":0.0})),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth(); // initial.size 10, lifetime 2
    p.age = 1.0; // lp 0.5 -> mult 0.5
    op.apply(&mut p, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
    approx(p.size, 5.0);
}

#[test]
fn alphachange_ramps_linearly() {
    let op = Operator::compile(
        &stage(json!({"name":"alphachange","startvalue":1.0,"endvalue":0.0})),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth();
    p.age = 1.5; // lp 0.75
    op.apply(&mut p, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
    approx(p.alpha, 0.25);
}

#[test]
fn colorchange_ramps_rgb_over_initial() {
    let op = Operator::compile(
        &stage(json!({"name":"colorchange","startvalue":"1 1 1","endvalue":"0 0 0"})),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth();
    p.age = 1.0; // lp 0.5
    op.apply(&mut p, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
    approx_v(p.color, [0.5, 0.5, 0.5]);
}

#[test]
fn controlpointattract_pulls_toward_cp() {
    let op = Operator::compile(
        &stage(json!({"name":"controlpointattract","controlpoint":0,"scale":100.0,"threshold":1000.0})),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth();
    p.position = [0.0, 0.0, 0.0];
    let cps = [[100.0, 0.0, 0.0]];
    op.apply(&mut p, &step_ctx(0.1, 0.0, &cps, &o));
    // dir toward (100,0,0) is +x; vel += dir*scale*dt = 100*0.1 = 10
    approx_v(p.velocity, [10.0, 0.0, 0.0]);
}

#[test]
fn controlpointattract_ignores_beyond_threshold() {
    let op = Operator::compile(
        &stage(json!({"name":"controlpointattract","scale":100.0,"threshold":100.0})),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth();
    p.position = [1000.0, 0.0, 0.0]; // beyond threshold/2 = 50
    let cps = [[0.0, 0.0, 0.0]];
    op.apply(&mut p, &step_ctx(0.1, 0.0, &cps, &o));
    approx_v(p.velocity, [0.0, 0.0, 0.0]);
}

#[test]
fn vortex_velocity_is_tangential() {
    let op = Operator::compile(
        &stage(json!({
            "name":"vortex","controlpoint":0,"axis":"0 0 1",
            "distanceinner":0.0,"distanceouter":1000.0,"speedinner":100.0,"speedouter":100.0,
            "centerforce":0.0
        })),
        1,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth();
    p.position = [10.0, 0.0, 0.0];
    let cps = [[0.0, 0.0, 0.0]];
    op.apply(&mut p, &step_ctx(0.1, 0.0, &cps, &o));
    // tangent = cross(z, radial=+x) = +y; radial component stays ~0.
    assert!(
        p.velocity[1] > 0.0,
        "expected +y tangential, got {:?}",
        p.velocity
    );
    approx(p.velocity[0], 0.0);
}

#[test]
fn turbulence_mask_zeroes_z() {
    let op = Operator::compile(
        &stage(json!({"name":"turbulence","mask":"1 1 0","speedmin":1000.0,"speedmax":1000.0})),
        7,
        &mut Rng::new(3),
    );
    let o = ovr();
    let mut p = synth();
    p.position = [5.0, 3.0, 1.0];
    op.apply(&mut p, &step_ctx(0.1, 0.5, &[[0.0; 3]], &o));
    approx(p.velocity[2], 0.0);
}

#[test]
fn turbulence_is_deterministic() {
    let mk = || Operator::compile(&stage(json!({"name":"turbulence"})), 7, &mut Rng::new(42));
    let o = ovr();
    let (mut a, mut b) = (synth(), synth());
    a.position = [2.0, -1.0, 0.5];
    b.position = [2.0, -1.0, 0.5];
    mk().apply(&mut a, &step_ctx(0.1, 1.0, &[[0.0; 3]], &o));
    mk().apply(&mut b, &step_ctx(0.1, 1.0, &[[0.0; 3]], &o));
    approx_v(a.velocity, b.velocity);
}

#[test]
fn oscillatesize_stays_within_scale_window() {
    let op = Operator::compile(
        &stage(
            json!({"name":"oscillatesize","frequencymin":1.0,"frequencymax":10.0,"scalemin":0.8,"scalemax":1.2}),
        ),
        11,
        &mut Rng::new(0),
    );
    let o = ovr();
    for age in [0.0, 0.3, 0.7, 1.3, 2.0] {
        let mut p = synth();
        p.size = 10.0;
        p.age = age;
        op.apply(&mut p, &step_ctx(0.0, 0.0, &[[0.0; 3]], &o));
        assert!(
            p.size >= 10.0 * 0.8 - 1e-3 && p.size <= 10.0 * 1.2 + 1e-3,
            "size={}",
            p.size
        );
    }
}

#[test]
fn oscillateposition_mask_disables_axis() {
    let op = Operator::compile(
        &stage(json!({"name":"oscillateposition","frequencymax":5.0,"scalemax":10.0,"mask":"1 0 0"})),
        13,
        &mut Rng::new(0),
    );
    let o = ovr();
    let mut p = synth();
    p.age = 0.4;
    op.apply(&mut p, &step_ctx(0.1, 0.0, &[[0.0; 3]], &o));
    approx(p.position[1], 0.0);
    approx(p.position[2], 0.0);
}

// ---- initializers ----------------------------------------------------------

fn spawn_ctx<'a>(cps: &'a [Vec3], o: &'a kirie_render::particle::Overrides) -> SpawnCtx<'a> {
    SpawnCtx {
        overrides: o,
        perspective: false,
        control_points: cps,
    }
}

#[test]
fn sizerandom_halves_and_applies_override() {
    // min == max removes randomness: size = 10 * sizeOverride / 2.
    let mut init = Initializer::compile(&stage(json!({"name":"sizerandom","min":10.0,"max":10.0})));
    let o = ovr();
    let mut p = synth();
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    approx(p.size, 5.0);
    approx(p.initial.size, 5.0);
}

#[test]
fn alpharandom_applies_override() {
    let mut init = Initializer::compile(&stage(json!({"name":"alpharandom","min":0.5,"max":0.5})));
    let mut o = ovr();
    o.alpha = 0.5;
    let mut p = synth();
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    approx(p.alpha, 0.25);
}

#[test]
fn lifetimerandom_applies_override() {
    let mut init = Initializer::compile(&stage(json!({"name":"lifetimerandom","min":2.0,"max":2.0})));
    let mut o = ovr();
    o.lifetime = 3.0;
    let mut p = synth();
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    approx(p.lifetime, 6.0);
}

#[test]
fn velocityrandom_flips_y_and_adds() {
    let mut init = Initializer::compile(&stage(
        json!({"name":"velocityrandom","min":"0 10 0","max":"0 10 0"}),
    ));
    let o = ovr();
    let mut p = synth();
    p.velocity = [1.0, 1.0, 0.0];
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    approx_v(p.velocity, [1.0, -9.0, 0.0]); // 1 + (-10)
}

#[test]
fn rotationrandom_sets_rotation() {
    let mut init = Initializer::compile(&stage(
        json!({"name":"rotationrandom","min":"0 0 1","max":"0 0 1"}),
    ));
    let o = ovr();
    let mut p = synth();
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    approx_v(p.rotation, [0.0, 0.0, 1.0]);
}

#[test]
fn colorrandom_multiplies_colorn() {
    let mut init = Initializer::compile(&stage(
        json!({"name":"colorrandom","min":"0.5 0.5 0.5","max":"0.5 0.5 0.5"}),
    ));
    let mut o = ovr();
    o.colorn = [1.0, 0.0, 0.5];
    let mut p = synth();
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    approx_v(p.color, [0.5, 0.0, 0.25]);
}

#[test]
fn mapsequence_round_robins_and_spawns_at_cp() {
    let mut init = Initializer::compile(&stage(json!({
        "name":"mapsequencearoundcontrolpoint","controlpoint":0,"count":4,
        "speedmin":"10 0 0","speedmax":"10 0 0"
    })));
    let o = ovr();
    let cps = [[7.0, 8.0, 9.0]];
    let mut rng = Rng::new(0);
    // Four spawns rotate the (10,0,0) base velocity by 0, 90, 180, 270 degrees.
    let mut vels = Vec::new();
    for _ in 0..4 {
        let mut p = synth();
        p.velocity = [0.0, 0.0, 0.0];
        init.apply(&mut p, &spawn_ctx(&cps, &o), &mut rng);
        approx_v(p.position, [7.0, 8.0, 9.0]);
        vels.push(p.velocity);
    }
    approx_v(vels[0], [10.0, 0.0, 0.0]);
    approx_v(vels[1], [0.0, 10.0, 0.0]);
    approx_v(vels[2], [-10.0, 0.0, 0.0]);
    approx_v(vels[3], [0.0, -10.0, 0.0]);
}

#[test]
fn unknown_stage_names_are_noops() {
    let mut init = Initializer::compile(&stage(json!({"name":"totallymadeup"})));
    let op = Operator::compile(&stage(json!({"name":"alsomadeup"})), 1, &mut Rng::new(0));
    let o = ovr();
    let before = synth();
    let mut p = synth();
    let mut rng = Rng::new(0);
    init.apply(&mut p, &spawn_ctx(&[[0.0; 3]], &o), &mut rng);
    op.apply(&mut p, &step_ctx(0.1, 0.0, &[[0.0; 3]], &o));
    assert_eq!(p, before);
}

// ---- emitter timing / sim --------------------------------------------------

fn sim_with(emitter: serde_json::Value, extra: serde_json::Value) -> ParticleSim {
    let mut def = serde_json::Map::new();
    def.insert("maxcount".into(), json!(1000));
    def.insert("emitter".into(), json!([emitter]));
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        def.insert(k, v);
    }
    let system = ParticleSystem::from_value(&serde_json::Value::Object(def));
    ParticleSim::new(
        &system,
        &InstanceOverride::default(),
        SimConfig { seed: 1, sheet: None },
    )
}

#[test]
fn boxrandom_rate_emits_one_per_tenth_second() {
    // rate 10, dt 0.1 -> 1 particle/step; long lifetimes so none die.
    let mut sim = sim_with(
        json!({"name":"boxrandom","rate":10.0}),
        json!({"initializer":[{"name":"lifetimerandom","min":100.0,"max":100.0}]}),
    );
    for _ in 0..10 {
        sim.update(0.1);
    }
    assert_eq!(sim.live_count(), 10);
    assert_eq!(sim.total_spawned(), 10);
}

#[test]
fn one_spawn_per_frame_flag_limits_emission() {
    // rate 100 would spawn 10/step; flags 0x2 caps to 1/step.
    let mut sim = sim_with(
        json!({"name":"boxrandom","rate":100.0,"flags":2}),
        json!({"initializer":[{"name":"lifetimerandom","min":100.0,"max":100.0}]}),
    );
    for _ in 0..5 {
        sim.update(0.1);
    }
    assert_eq!(sim.live_count(), 5);
}

#[test]
fn instantaneous_emits_single_burst() {
    // instantaneous -> emit round(rate) once, then nothing more.
    let mut sim = sim_with(
        json!({"name":"boxrandom","rate":8.0,"instantaneous":1}),
        json!({"initializer":[{"name":"lifetimerandom","min":100.0,"max":100.0}]}),
    );
    sim.update(0.1);
    assert_eq!(sim.total_spawned(), 8);
    for _ in 0..10 {
        sim.update(0.1);
    }
    assert_eq!(sim.total_spawned(), 8, "burst must not repeat");
}

#[test]
fn emitter_delay_gates_start() {
    let mut sim = sim_with(
        json!({"name":"boxrandom","rate":10.0,"delay":1.0}),
        json!({"initializer":[{"name":"lifetimerandom","min":100.0,"max":100.0}]}),
    );
    for _ in 0..5 {
        sim.update(0.1); // t up to 0.5 < delay
    }
    assert_eq!(sim.total_spawned(), 0);
    for _ in 0..10 {
        sim.update(0.1); // crosses t=1.0
    }
    assert!(sim.total_spawned() > 0, "should emit after the delay");
}

#[test]
fn lifetime_culls_particles() {
    // Emit continuously with a 1s lifetime; live count converges to rate*lifetime.
    let mut sim = sim_with(
        json!({"name":"boxrandom","rate":10.0}),
        json!({"initializer":[{"name":"lifetimerandom","min":1.0,"max":1.0}]}),
    );
    for _ in 0..300 {
        sim.update(1.0 / 30.0);
    }
    let live = sim.live_count();
    // 10 particles/s spawned, each living 1s -> ~10 alive (± a frame's worth).
    assert!((9..=11).contains(&live), "live={live}");
}

#[test]
fn pool_capacity_is_never_exceeded() {
    let mut def = serde_json::Map::new();
    def.insert("maxcount".into(), json!(5));
    def.insert("emitter".into(), json!([{"name":"boxrandom","rate":1000.0}]));
    def.insert(
        "initializer".into(),
        json!([{"name":"lifetimerandom","min":100.0,"max":100.0}]),
    );
    let system = ParticleSystem::from_value(&serde_json::Value::Object(def));
    let mut sim = ParticleSim::new(
        &system,
        &InstanceOverride::default(),
        SimConfig { seed: 1, sheet: None },
    );
    assert_eq!(sim.capacity(), 5);
    for _ in 0..50 {
        sim.update(0.1);
        assert!(sim.live_count() <= 5);
    }
    assert_eq!(sim.live_count(), 5);
}

#[test]
fn dt_is_capped_at_max_dt() {
    let mut sim = sim_with(
        json!({"name":"boxrandom","rate":10.0}),
        json!({"initializer":[{"name":"lifetimerandom","min":100.0,"max":100.0}]}),
    );
    // A huge dt advances time by at most MAX_DT (0.1) -> 1 spawn, not thousands.
    sim.update(1000.0);
    assert_eq!(sim.total_spawned(), 1);
    approx(sim.time(), kirie_render::particle::MAX_DT);
}

#[test]
fn unsupported_emitter_spawns_nothing() {
    let sim = sim_with(json!({"name":"coolrandom","rate":10.0}), json!({}));
    assert!(!sim.has_supported_emitter());
}

#[test]
fn instance_override_count_scales_pool() {
    let system = ParticleSystem::from_value(&json!({"maxcount":100}));
    let ov = InstanceOverride {
        count: kirie_scene::UserSetting::literal(2.0),
        ..InstanceOverride::default()
    };
    let sim = ParticleSim::new(&system, &ov, SimConfig::default());
    assert_eq!(sim.capacity(), 200);
}

#[test]
fn disabled_override_freezes_sim() {
    let ov = InstanceOverride {
        enabled: kirie_scene::UserSetting::literal(false),
        ..InstanceOverride::default()
    };
    let system = ParticleSystem::from_value(&json!({
        "maxcount":100,"emitter":[{"name":"boxrandom","rate":10.0}]
    }));
    let mut sim = ParticleSim::new(&system, &ov, SimConfig::default());
    for _ in 0..10 {
        sim.update(0.1);
    }
    assert_eq!(sim.total_spawned(), 0);
}
