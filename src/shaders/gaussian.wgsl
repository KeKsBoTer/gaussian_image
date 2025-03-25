const CUT_OFF:f32 = 1./255.;

struct Splat {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) offset: vec2<f32>,
    @location(3) cov: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) offset: vec2<f32>,
    @location(2) cov: vec4<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) dx: vec4<f32>,
    @location(2) dy: vec4<f32>,
    @location(3) dxy: vec4<f32>,
};

struct Settings{
    scale_factor:f32,
    channel:u32,
    upscale_factor:f32,
    upscaling_method:u32,
    clamp_gradients:u32,
    clamp_image:u32,
    _pad0:u32,
    _pad1:u32,
}

@group(0) @binding(0)
var<uniform> settings : Settings;

@vertex
fn vs_main(
    splat:Splat,
) -> VertexOutput {
    return VertexOutput(
        vec4<f32>(splat.position.xy,0.,1.),
        splat.color,
        splat.offset / settings.scale_factor,
        splat.cov
    );
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let delta = in.offset;
    let cov = mat2x2<f32>(in.cov.xy, in.cov.zw);
    let power = -0.5 * (cov[0][0] * delta.x * delta.x +
                                        cov[1][1] * delta.y*delta.y) +
                                cov[0][1] * delta.x * delta.y;

    let alpha = exp(power);
    let alpha_clamped = min(1., alpha);

    if alpha_clamped < CUT_OFF {
        discard;
    }

    let x = delta.x;
    let y = delta.y;

    let c_a = cov[0][0];
    let c_b = cov[1][0];
    let c_c = cov[1][1];
    let c_d = cov[0][1];

    let dg_dx = (- c_a * x + c_b * y);
    let dg_dy = (- c_b * x + c_c * y);
    let dg_dxy = -c_b;

    var alpha_dx = dg_dx * alpha;
    var alpha_dy = dg_dy * alpha;
    var alpha_dxy = (dg_dx * dg_dy+ dg_dxy ) * alpha;

    if alpha !=alpha_clamped{
        alpha_dx = 0.;
        alpha_dy = 0.;
        alpha_dxy = 0.;
    }

    let color = in.color.rgb;

    var frag_out:FragmentOutput;

    frag_out.color = vec4<f32>(color*alpha_clamped, 1.);
    frag_out.dx = vec4<f32>(color*alpha_dx, 1.);
    frag_out.dy = vec4<f32>(color*alpha_dy, 1.);
    frag_out.dxy = vec4<f32>(color*alpha_dxy, 1.);
    return frag_out;
}
