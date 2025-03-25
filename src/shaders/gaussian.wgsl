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
    @location(0) image_0: vec4<f32>,
    @location(1) image_1: vec4<f32>,
    @location(2) image_2: vec4<f32>,
    @location(3) image_3: vec4<f32>,
    @location(4) image_4: vec4<f32>,
    @location(5) image_5: vec4<f32>,
    @location(6) image_6: vec4<f32>,
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
    let dg_dy = (c_b * x - c_c * y);
    let dg_dxy = c_b;
    let dg_dxx = -c_a;
    let dg_dyy = -c_c;

    let dg_dxxy = 0.;
    let dg_dxyy = 0.;
    let dg_dxxyy = 0.;

    var alpha_dx = dg_dx * alpha;
    var alpha_dy = dg_dy * alpha;
    var alpha_dxy = (dg_dx * dg_dy+ dg_dxy ) * alpha;
    var alpha_dxx = (dg_dx*dg_dx+dg_dxx)*alpha;
    var alpha_dyy = (dg_dy*dg_dy+dg_dyy)*alpha;
    var alpha_dxxy = (dg_dx*dg_dx*dg_dy+2.*dg_dx*dg_dxy+dg_dxx*dg_dy+dg_dxxy)*alpha;
    var alpha_dxyy = (dg_dx*dg_dy*dg_dy+2.*dg_dy*dg_dxy+dg_dx*dg_dyy+dg_dxyy)*alpha;
    var alpha_dxxyy = (dg_dx*dg_dx*dg_dy*dg_dy+dg_dx*dg_dx*dg_dyy+4.*dg_dx*dg_dy*dg_dxy+2.*dg_dx*dg_dxyy+dg_dxx*dg_dy*dg_dy+dg_dxx*dg_dyy+2*dg_dy*dg_dxxy+2*dg_dxy*dg_dxy+dg_dxxyy)*alpha;

    if alpha !=alpha_clamped{
        alpha_dx = 0.;
        alpha_dy = 0.;
        alpha_dxy = 0.;
        alpha_dxx = 0.;
        alpha_dyy = 0.;
        alpha_dxxy = 0.;
        alpha_dxyy = 0.;
        alpha_dxxyy = 0.;
    }

    let color = in.color.rgb;

    var frag_out:FragmentOutput;

    let color_c = color*alpha_clamped;
    let dx = color*alpha_dx;
    let dy = color*alpha_dy;
    let dxy = color*alpha_dxy;
    let dxx = color*alpha_dxx;
    let dyy = color*alpha_dyy;
    let dxxy = color*alpha_dxxy;
    let dxyy = color*alpha_dxyy;
    let dxxyy = color*alpha_dxxyy;

    frag_out.image_0 = vec4<f32>(color_c,dx.r);
    frag_out.image_1 = vec4<f32>(dx.gb,dy.rg);
    frag_out.image_2 = vec4<f32>(dy.b,dxy.rgb);
    frag_out.image_3 = vec4<f32>(dxx.rgb,dyy.r);
    frag_out.image_4 = vec4<f32>(dyy.gb,dxxy.rg);
    frag_out.image_5 = vec4<f32>(dxxy.b,dxyy.rgb);
    frag_out.image_6 = vec4<f32>(dxxyy.rgb,0.);
    
    return frag_out;
}
