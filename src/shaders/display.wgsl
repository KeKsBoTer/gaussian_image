const CHANNEL_COLOR:u32 = 0u;
const CHANNEL_DX:u32 = 1u;
const CHANNEL_DY:u32 = 2u;
const CHANNEL_DXY:u32 = 3u;

const INTERPOLATION_NEAREST:u32 = 0u;
const INTERPOLATION_BILINEAR:u32 = 1u;
const INTERPOLATION_BICUBIC:u32 = 2u;
const INTERPOLATION_SPLINE:u32 = 3u;

@group(0) @binding(0)
var<uniform> settings : Settings;

@group(1) @binding(0)
var source_img : texture_2d<f32>;

@group(1) @binding(1)
var image_dx : texture_2d<f32>;

@group(1) @binding(2)
var image_dy : texture_2d<f32>;

@group(1) @binding(3)
var image_dxy : texture_2d<f32>;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}


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

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOut {

    // creates two vertices that cover the whole screen
    let xy = vec2<f32>(
        f32(in_vertex_index % 2u == 0u),
        f32(in_vertex_index < 2u)
    );
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), vec2<f32>(xy.x, xy.y));
}


@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> 
{
    let tex_size = textureSize(source_img);
    let pixel_pos = vec2<i32>(tex_size * vertex_in.tex_coord);
    var grad:f32;
    switch settings.channel{
        case CHANNEL_DX{
            grad = textureLoad_scaled(image_dx, pixel_pos).r*10.;
        }
        case CHANNEL_DY{
            grad = textureLoad_scaled(image_dy, pixel_pos).r*10.;
        }
        case CHANNEL_DXY{
            grad = textureLoad_scaled(image_dxy, pixel_pos).r*10.;
        }
        default{
             switch settings.upscaling_method{
                case INTERPOLATION_BILINEAR:{
                    return sample_bilinear(vertex_in.tex_coord);
                }
                case INTERPOLATION_BICUBIC:{
                    return sample_bicubic(vertex_in.tex_coord);
                }
                case INTERPOLATION_SPLINE:{
                    return sample_spline(vertex_in.tex_coord);
                }
                default:{
                    return sample_nearest(vertex_in.tex_coord);
                }
            }
        }
    }

    let color_min = vec4<f32>(0.,0.,1.,1.);
    let color_center = vec4<f32>(0.,0.,0.,1.);
    let color_max = vec4<f32>(1.,0.,0.,1.);
    if grad < 0.{
        return mix(color_center, color_min, -grad);
    }
    return mix(color_center, color_max, grad);
}


fn textureSize(tex: texture_2d<f32>) -> vec2<f32> {
    return vec2<f32>(textureDimensions(tex,0))/settings.upscale_factor;
}

fn textureLoad_scaled(tex: texture_2d<f32>, pos: vec2<i32>) -> vec4<f32> {
    return textureLoad(tex, 
        clamp(vec2<i32>(vec2<f32>(pos)*settings.upscale_factor), vec2<i32>(0), vec2<i32>(textureDimensions(tex)-1))
    , 0);
}

fn sample_nearest(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = textureSize(source_img);
    let pos = pos_in*(tex_size)+0.5;
    let pixel_pos = vec2<i32>(pos);
    return textureLoad_scaled(source_img, pixel_pos);
}


fn sample_bilinear(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = textureSize(source_img);

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);

    let z00 = textureLoad_scaled(source_img, pixel_pos+vec2<i32>(0,0));
    let z10 = textureLoad_scaled(source_img, pixel_pos+vec2<i32>(1,0));
    let z01 = textureLoad_scaled(source_img, pixel_pos+vec2<i32>(0,1));
    let z11 = textureLoad_scaled(source_img, pixel_pos+vec2<i32>(1,1));

    return mix(
        mix(z00, z10, p_frac.x),
        mix(z01, z11, p_frac.x),
        p_frac.y
    );
}


fn sample_spline(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = textureSize(source_img);

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = pixel_pos + vec2<i32>(i, j);

            let s = 1.*settings.upscale_factor;
            var z_v = textureLoad_scaled(source_img, sample_pos);
            var dx_v = textureLoad_scaled(image_dx, sample_pos)*s;
            var dy_v = textureLoad_scaled(image_dy, sample_pos)*s;
            var dxy_v = textureLoad_scaled(image_dxy, sample_pos)*s*s;

            if bool(settings.clamp_gradients){
                dx_v = clamp(dx_v, vec4<f32>(-1.), vec4<f32>(1.));
                dy_v = clamp(dy_v, vec4<f32>(-1.), vec4<f32>(1.));
                dxy_v = clamp(dxy_v, vec4<f32>(-1.), vec4<f32>(1.));
            }
            
            // clamping image values and gradients since image can have values outside of [0,1]
            if bool(settings.clamp_image){
                for(var c = 0u; c < 4u; c++){
                    if z_v[c] < 0. || z_v[c] > 1.{
                        z_v[c] = clamp(z_v[c],0.,1.);
                        dx_v[c] = 0.;
                        dy_v[c] = 0.;
                        dxy_v[c] = 0.;
                    }
                }
            }



            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = dx_v[c];
                dy[c][i][j] = dy_v[c];
                dxy[c][i][j] = dxy_v[c];
            }

        }
    }

    return vec4<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[3], dx[3], dy[3], dxy[3], vec2<f32>(p_frac.y, p_frac.x))
    ); 
}


fn sample_bicubic(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = textureSize(source_img);

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = pixel_pos + vec2<i32>(i, j);

            var z_v = textureLoad_scaled(source_img, sample_pos);
            
            var z_left_down = textureLoad_scaled(source_img, sample_pos+vec2<i32>(-1,-1));
            var z_left_mid = textureLoad_scaled(source_img, sample_pos+vec2<i32>(-1,0));
            var z_left_up = textureLoad_scaled(source_img, sample_pos+vec2<i32>(-1,1));
            var z_right_down = textureLoad_scaled(source_img, sample_pos+vec2<i32>(1,-1));
            var z_right_mid = textureLoad_scaled(source_img, sample_pos+vec2<i32>(1,0));
            var z_right_up = textureLoad_scaled(source_img, sample_pos+vec2<i32>(1,1));
            var z_mid_up = textureLoad_scaled(source_img, sample_pos+vec2<i32>(0,1));
            var z_mid_down = textureLoad_scaled(source_img, sample_pos+vec2<i32>(0,-1));

            if bool(settings.clamp_image){
                z_v = clamp(z_v, vec4<f32>(0.), vec4<f32>(1.));
                z_left_down = clamp(z_left_down, vec4<f32>(0.), vec4<f32>(1.));
                z_left_mid = clamp(z_left_mid, vec4<f32>(0.), vec4<f32>(1.));
                z_left_up = clamp(z_left_up, vec4<f32>(0.), vec4<f32>(1.));
                z_right_down = clamp(z_right_down, vec4<f32>(0.), vec4<f32>(1.));
                z_right_mid = clamp(z_right_mid, vec4<f32>(0.), vec4<f32>(1.));
                z_right_up = clamp(z_right_up, vec4<f32>(0.), vec4<f32>(1.));
                z_mid_up = clamp(z_mid_up, vec4<f32>(0.), vec4<f32>(1.));
                z_mid_down = clamp(z_mid_down, vec4<f32>(0.), vec4<f32>(1.));
            }
            

            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = (z_right_mid[c] - z_left_mid[c]) * 0.5;
                dy[c][i][j] = (z_mid_up[c] - z_mid_down[c]) * 0.5;
                dxy[c][i][j] = (z_right_up[c] - z_right_down[c] - z_left_up[c] + z_left_down[c]) * 0.25;
            }

        }
    }

    return vec4<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[3], dx[3], dy[3], dxy[3], vec2<f32>(p_frac.y, p_frac.x))
    ); 
}

fn spline_interp( z:mat2x2<f32>, dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>, p: vec2<f32>) -> f32
{
    let f = mat4x4<f32>(
        z[0][0], z[0][1], dy[0][0], dy[0][1],
        z[1][0], z[1][1], dy[1][0], dy[1][1],
        dx[0][0], dx[0][1], dxy[0][0], dxy[0][1],
        dx[1][0], dx[1][1], dxy[1][0], dxy[1][1]
    );
    let m = mat4x4<f32>(
        1., 0., 0., 0.,
        0., 0., 1., 0.,
        -3., 3., -2., -1.,
        2., -2., 1., 1.
    );
    let a = transpose(m) * f * (m);

    let tx = vec4<f32>(1., p.x, p.x * p.x, p.x * p.x * p.x);
    let ty = vec4<f32>(1., p.y, p.y * p.y, p.y * p.y * p.y);
    return dot(tx, a * ty);
}

