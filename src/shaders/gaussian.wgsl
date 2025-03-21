const CUTOFF:f32 = sqrt(log(255.));

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



@vertex
fn vs_main(
    splat:Splat,
) -> VertexOutput {
    return VertexOutput(
        vec4<f32>((splat.position.xy)/vec2<f32>(768.,512.)*2-1.,0.,1.),
        splat.color,
        splat.offset,
        splat.cov
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let delta = in.offset;
    let cov = mat2x2<f32>(in.cov.xy, in.cov.zw);
    let power = -0.5 * (cov[0][0] * delta.x * delta.x +
                                        cov[1][1] * delta.y*delta.y) +
                                cov[0][1] * delta.x * delta.y;

    let alpha = exp(power);
    let alpha_clamped = min(1., alpha);
    let color = in.color.rgb;

    return vec4<f32>(color*alpha_clamped, 1.);
}
