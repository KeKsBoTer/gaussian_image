const INTERPOLATION_NEAREST:u32 = 0u;
const INTERPOLATION_BILINEAR:u32 = 1u;
const INTERPOLATION_BICUBIC:u32 = 2u;
const INTERPOLATION_SPLINE3:u32 = 3u;
const INTERPOLATION_SPLINE5:u32 = 4u;

@group(0) @binding(0)
var<uniform> settings : Settings;

@group(1) @binding(0)
var image_0 : texture_2d<f32>;

@group(1) @binding(1)
var image_1 : texture_2d<f32>;

@group(1) @binding(2)
var image_2 : texture_2d<f32>;

@group(1) @binding(3)
var image_3 : texture_2d<f32>;

@group(1) @binding(4)
var image_4 : texture_2d<f32>;

@group(1) @binding(5)
var image_5 : texture_2d<f32>;

@group(1) @binding(6)
var image_6 : texture_2d<f32>;



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
    let tex_size = textureSize();
    let pixel_pos = vec2<i32>(tex_size * vertex_in.tex_coord);
    var grad:f32;
    switch settings.channel{
        case 1u{
            grad = textureLoad_scaled(1u, pixel_pos).r*10.;
        }
        case 2u{
            grad = textureLoad_scaled(2u, pixel_pos).r*10.;
        }
        case 3u{
            grad = textureLoad_scaled(3u, pixel_pos).r*10.;
        }
        case 4u{
            grad = textureLoad_scaled(4u, pixel_pos).r*10.;
        }
        case 5u{
            grad = textureLoad_scaled(5u, pixel_pos).r*10.;
        }
        case 6u{
            grad = textureLoad_scaled(6u, pixel_pos).r*10.;
        }
        case 7u{
            grad = textureLoad_scaled(7u, pixel_pos).r*10.;
        }
        case 8u{
            grad = textureLoad_scaled(8u, pixel_pos).r*10.;
        }
        default{
             switch settings.upscaling_method{
                case INTERPOLATION_BILINEAR:{
                    return vec4<f32>(sample_bilinear(vertex_in.tex_coord),1.);
                }
                case INTERPOLATION_BICUBIC:{
                    return vec4<f32>(sample_bicubic(vertex_in.tex_coord),1.);
                }
                case INTERPOLATION_SPLINE3:{
                    return vec4<f32>(sample_spline3(vertex_in.tex_coord),1.);
                }
                case INTERPOLATION_SPLINE5:{
                    return vec4<f32>(sample_spline5(vertex_in.tex_coord),1.);
                }
                default:{
                    return vec4<f32>(sample_nearest(vertex_in.tex_coord),1.);
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


fn textureSize() -> vec2<f32> {
    return vec2<f32>(textureDimensions(image_0,0))/settings.upscale_factor;
}

fn textureLoad_scaled(tex: u32, pos: vec2<i32>) -> vec3<f32> {
    let pos_c =clamp(vec2<i32>(vec2<f32>(pos)*settings.upscale_factor), vec2<i32>(0), vec2<i32>(textureDimensions(image_0)-1));
    switch tex{
        case 0u:{
            return textureLoad(image_0, pos_c, 0).rgb;
        }
        case 1u:{
            let a = textureLoad(image_0, pos_c, 0).a;
            let b = textureLoad(image_1, pos_c, 0).rg;
            return vec3<f32>(a,b);
        }
        case 2u:{
            let a = textureLoad(image_1, pos_c, 0).ba;
            let b = textureLoad(image_2, pos_c, 0).r;
            return vec3<f32>(a,b);
        }
        case 3u:{
            return textureLoad(image_2, pos_c, 0).gba;
        }
        case 4u:{
            return textureLoad(image_3, pos_c, 0).rgb;
        }
        case 5u:{
            let a = textureLoad(image_3, pos_c, 0).a;
            let b = textureLoad(image_4, pos_c, 0).rg;
            return vec3<f32>(a,b);
        }
        case 6u:{
            let a = textureLoad(image_4, pos_c, 0).ba;
            let b = textureLoad(image_5, pos_c, 0).r;
            return vec3<f32>(a,b);
        }
        case 7u:{
            return textureLoad(image_5, pos_c, 0).gba;
        }
        case 8u:{
            return textureLoad(image_6, pos_c, 0).rgb;
        }
        default:{
            return vec3<f32>(0.);
        }
    }
    
}

fn sample_nearest(pos_in:vec2<f32>)->vec3<f32>{
    let tex_size = textureSize();
    let pos = pos_in*(tex_size)+0.5;
    let pixel_pos = vec2<i32>(pos);
    return textureLoad_scaled(0u, pixel_pos);
}


fn sample_bilinear(pos_in:vec2<f32>)->vec3<f32>{
    let tex_size = textureSize();

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);

    let z00 = textureLoad_scaled(0u, pixel_pos+vec2<i32>(0,0));
    let z10 = textureLoad_scaled(0u, pixel_pos+vec2<i32>(1,0));
    let z01 = textureLoad_scaled(0u, pixel_pos+vec2<i32>(0,1));
    let z11 = textureLoad_scaled(0u, pixel_pos+vec2<i32>(1,1));

    return mix(
        mix(z00, z10, p_frac.x),
        mix(z01, z11, p_frac.x),
        p_frac.y
    );
}


fn sample_spline3(pos_in:vec2<f32>)->vec3<f32>{
    let tex_size = textureSize();

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,3>;
    var dx: array<mat2x2<f32>,3>;
    var dy: array<mat2x2<f32>,3>;
    var dxy:array<mat2x2<f32>,3>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = pixel_pos + vec2<i32>(i, j);

            let s = 1.*settings.upscale_factor;
            var z_v = textureLoad_scaled(0u, sample_pos);
            var dx_v = textureLoad_scaled(1u, sample_pos)*s;
            var dy_v = -textureLoad_scaled(2u, sample_pos)*s;
            var dxy_v = -textureLoad_scaled(3u, sample_pos)*s*s;

            if bool(settings.clamp_gradients){
                dx_v = clamp(dx_v, vec3<f32>(-1.), vec3<f32>(1.));
                dy_v = clamp(dy_v, vec3<f32>(-1.), vec3<f32>(1.));
                dxy_v = clamp(dxy_v, vec3<f32>(-1.), vec3<f32>(1.));
            }
            
            // clamping image values and gradients since image can have values outside of [0,1]
            if bool(settings.clamp_image){
                for(var c = 0u; c < 3u; c++){
                    if z_v[c] < 0. || z_v[c] > 1.{
                        z_v[c] = clamp(z_v[c],0.,1.);
                        dx_v[c] = 0.;
                        dy_v[c] = 0.;
                        dxy_v[c] = 0.;
                    }
                }
            }



            for (var c = 0u; c < 3u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = dx_v[c];
                dy[c][i][j] = dy_v[c];
                dxy[c][i][j] = dxy_v[c];
            }

        }
    }

    return vec3<f32>(
        spline_interp3(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp3(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp3(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
    ); 
}

fn sample_spline5(pos_in:vec2<f32>)->vec3<f32>{
    let tex_size = textureSize();

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,3>;
    var dx: array<mat2x2<f32>,3>;
    var dy: array<mat2x2<f32>,3>;
    var dxy:array<mat2x2<f32>,3>;
    var dxx: array<mat2x2<f32>,3>;
    var dyy: array<mat2x2<f32>,3>;
    var dxxy:array<mat2x2<f32>,3>;
    var dxyy:array<mat2x2<f32>,3>;
    var dxxyy:array<mat2x2<f32>,3>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = pixel_pos + vec2<i32>(i, j);

            let s = 1.*settings.upscale_factor;
            var z_v = textureLoad_scaled(0u, sample_pos);
            var dx_v = textureLoad_scaled(1u, sample_pos)*s;
            var dy_v = -textureLoad_scaled(2u, sample_pos)*s;
            var dxy_v = -textureLoad_scaled(3u, sample_pos)*s*s;
            var dxx_v = textureLoad_scaled(4u, sample_pos)*s*s;
            var dyy_v = -textureLoad_scaled(5u, sample_pos)*s*s;
            var dxxy_v = -textureLoad_scaled(6u, sample_pos)*s*s*s;
            var dxyy_v = -textureLoad_scaled(7u, sample_pos)*s*s*s;
            var dxxyy_v = textureLoad_scaled(8u, sample_pos)*s*s*s*s;

            if bool(settings.clamp_gradients){
                dx_v = clamp(dx_v, vec3<f32>(-1.), vec3<f32>(1.));
                dy_v = clamp(dy_v, vec3<f32>(-1.), vec3<f32>(1.));
                dxy_v = clamp(dxy_v, vec3<f32>(-1.), vec3<f32>(1.));
                dxx_v = clamp(dxx_v, vec3<f32>(-1.), vec3<f32>(1.));
                dyy_v = clamp(dyy_v, vec3<f32>(-1.), vec3<f32>(1.));
                dxxy_v = clamp(dxxy_v, vec3<f32>(-1.), vec3<f32>(1.));
                dxyy_v = clamp(dxyy_v, vec3<f32>(-1.), vec3<f32>(1.));
                dxxyy_v = clamp(dxxyy_v, vec3<f32>(-1.), vec3<f32>(1.));
            }
            
            // clamping image values and gradients since image can have values outside of [0,1]
            if bool(settings.clamp_image){
                for(var c = 0u; c < 3u; c++){
                    if z_v[c] < 0. || z_v[c] > 1.{
                        z_v[c] = clamp(z_v[c],0.,1.);
                        dx_v[c] = 0.;
                        dy_v[c] = 0.;
                        dxy_v[c] = 0.;
                        dxx_v[c] = 0.;
                        dyy_v[c] = 0.;
                        dxxy_v[c] = 0.;
                        dxyy_v[c] = 0.;
                        dxxyy_v[c] = 0.;
                    }
                }
            }



            for (var c = 0u; c < 3u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = dx_v[c];
                dy[c][i][j] = dy_v[c];
                dxy[c][i][j] = dxy_v[c];
                dxx[c][i][j] = dxx_v[c];
                dyy[c][i][j] = dyy_v[c];
                dxxy[c][i][j] = dxxy_v[c];
                dxyy[c][i][j] = dxyy_v[c];
                dxxyy[c][i][j] = dxxyy_v[c];
            }

        }
    }

    return vec3<f32>(
        spline_interp5(z[0], dx[0], dy[0], dxy[0], dxx[0], dyy[0], dxxy[0], dxyy[0], dxxyy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp5(z[1], dx[1], dy[1], dxy[1], dxx[1], dyy[1], dxxy[1], dxyy[1], dxxyy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp5(z[2], dx[2], dy[2], dxy[2], dxx[2], dyy[2], dxxy[2], dxyy[2], dxxyy[2], vec2<f32>(p_frac.y, p_frac.x)),
    ); 
}


fn sample_bicubic(pos_in:vec2<f32>)->vec3<f32>{
    let tex_size = textureSize();

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,3>;
    var dx: array<mat2x2<f32>,3>;
    var dy: array<mat2x2<f32>,3>;
    var dxy:array<mat2x2<f32>,3>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = pixel_pos + vec2<i32>(i, j);

            var z_v = textureLoad_scaled(0u, sample_pos);
            
            var z_left_down = textureLoad_scaled(0u, sample_pos+vec2<i32>(-1,-1));
            var z_left_mid = textureLoad_scaled(0u, sample_pos+vec2<i32>(-1,0));
            var z_left_up = textureLoad_scaled(0u, sample_pos+vec2<i32>(-1,1));
            var z_right_down = textureLoad_scaled(0u, sample_pos+vec2<i32>(1,-1));
            var z_right_mid = textureLoad_scaled(0u, sample_pos+vec2<i32>(1,0));
            var z_right_up = textureLoad_scaled(0u, sample_pos+vec2<i32>(1,1));
            var z_mid_up = textureLoad_scaled(0u, sample_pos+vec2<i32>(0,1));
            var z_mid_down = textureLoad_scaled(0u, sample_pos+vec2<i32>(0,-1));

            if bool(settings.clamp_image){
                z_v = clamp(z_v, vec3<f32>(0.), vec3<f32>(1.));
                z_left_down = clamp(z_left_down, vec3<f32>(0.), vec3<f32>(1.));
                z_left_mid = clamp(z_left_mid, vec3<f32>(0.), vec3<f32>(1.));
                z_left_up = clamp(z_left_up, vec3<f32>(0.), vec3<f32>(1.));
                z_right_down = clamp(z_right_down, vec3<f32>(0.), vec3<f32>(1.));
                z_right_mid = clamp(z_right_mid, vec3<f32>(0.), vec3<f32>(1.));
                z_right_up = clamp(z_right_up, vec3<f32>(0.), vec3<f32>(1.));
                z_mid_up = clamp(z_mid_up, vec3<f32>(0.), vec3<f32>(1.));
                z_mid_down = clamp(z_mid_down, vec3<f32>(0.), vec3<f32>(1.));
            }
            

            for (var c = 0u; c < 3u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = (z_right_mid[c] - z_left_mid[c]) * 0.5;
                dy[c][i][j] = (z_mid_up[c] - z_mid_down[c]) * 0.5;
                dxy[c][i][j] = (z_right_up[c] - z_right_down[c] - z_left_up[c] + z_left_down[c]) * 0.25;
            }

        }
    }

    return vec3<f32>(
        spline_interp3(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp3(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp3(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
    ); 
}

fn spline_interp3( z:mat2x2<f32>, dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>, p: vec2<f32>) -> f32
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


fn spline_interp5( z:mat2x2<f32>,
    dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>,
    dxx:mat2x2<f32>, dyy: mat2x2<f32>, dxxy: mat2x2<f32>,
    dxyy:mat2x2<f32>, dxxyy: mat2x2<f32>, p: vec2<f32>) -> f32
{
    let f = array<f32,36>(
        z[0][0], z[0][1], dy[0][0], dy[0][1],dyy[0][0],dyy[0][1],
        z[1][0], z[1][1], dy[1][0], dy[1][1],dyy[1][0],dyy[1][1],
        dx[0][0], dx[0][1], dxy[0][0], dxy[0][1],dxyy[0][0],dxyy[0][1],
        dx[1][0], dx[1][1], dxy[1][0], dxy[1][1],dxyy[1][0],dxyy[1][1],
        dxx[0][0], dxx[0][1], dxxy[0][0], dxxy[0][1],dxxyy[0][0],dxxyy[0][1],
        dxx[1][0], dxx[1][1], dxxy[1][0], dxxy[1][1],dxxyy[1][0],dxxyy[1][1]
    );
    let m = array<f32,36>(
        1., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 1.0/2.0, 0.,
        -10., 10., -6., -4., -3.0/2.0, 1.0/2.0,
        15., -15., 8., 7., 3.0/2.0, -1.,
        -6., 6., -3., -3., -1.0/2.0, 1.0/2.0,
    );
    return spline_interp_calc5(m,f,p.x,p.y);
}


/// Calculate the value of the spline at a given point
/// does the following: 
/// a = transpose(m) * f * (m)
/// tx = [1., x,pow(x,2),pow(x,3),pow(x,4),pow(x,5)]
/// ty = [1., y,pow(y,2),pow(y,3),pow(y,4),pow(y,5)]
/// return dot(tx, a * ty)
/// code was generated with sympy as it is too long to write by hand and not performant when using loops
fn spline_interp_calc5(m:array<f32,36>,f:array<f32,36>,x:f32,y:f32)->f32{
    return 1.0*pow(x, 5.0)*((f[0]*m[30] + f[1]*m[31] + f[2]*m[32] + f[3]*m[33] + f[4]*m[34] + f[5]*m[35])*m[0] + (f[6]*m[30] + f[7]*m[31] + f[8]*m[32] + f[9]*m[33] + f[10]*m[34] + f[11]*m[35])*m[1] + (f[12]*m[30] + f[13]*m[31] + f[14]*m[32] + f[15]*m[33] + f[16]*m[34] + f[17]*m[35])*m[2] + (f[18]*m[30] + f[19]*m[31] + f[20]*m[32] + f[21]*m[33] + f[22]*m[34] + f[23]*m[35])*m[3] + (f[24]*m[30] + f[25]*m[31] + f[26]*m[32] + f[27]*m[33] + f[28]*m[34] + f[29]*m[35])*m[4] + (f[30]*m[30] + f[31]*m[31] + f[32]*m[32] + f[33]*m[33] + f[34]*m[34] + f[35]*m[35])*m[5]) + 1.0*pow(x, 4.0)*((f[0]*m[24] + f[1]*m[25] + f[2]*m[26] + f[3]*m[27] + f[4]*m[28] + f[5]*m[29])*m[0] + (f[6]*m[24] + f[7]*m[25] + f[8]*m[26] + f[9]*m[27] + f[10]*m[28] + f[11]*m[29])*m[1] + (f[12]*m[24] + f[13]*m[25] + f[14]*m[26] + f[15]*m[27] + f[16]*m[28] + f[17]*m[29])*m[2] + (f[18]*m[24] + f[19]*m[25] + f[20]*m[26] + f[21]*m[27] + f[22]*m[28] + f[23]*m[29])*m[3] + (f[24]*m[24] + f[25]*m[25] + f[26]*m[26] + f[27]*m[27] + f[28]*m[28] + f[29]*m[29])*m[4] + (f[30]*m[24] + f[31]*m[25] + f[32]*m[26] + f[33]*m[27] + f[34]*m[28] + f[35]*m[29])*m[5]) + 1.0*pow(x, 3.0)*((f[0]*m[18] + f[1]*m[19] + f[2]*m[20] + f[3]*m[21] + f[4]*m[22] + f[5]*m[23])*m[0] + (f[6]*m[18] + f[7]*m[19] + f[8]*m[20] + f[9]*m[21] + f[10]*m[22] + f[11]*m[23])*m[1] + (f[12]*m[18] + f[13]*m[19] + f[14]*m[20] + f[15]*m[21] + f[16]*m[22] + f[17]*m[23])*m[2] + (f[18]*m[18] + f[19]*m[19] + f[20]*m[20] + f[21]*m[21] + f[22]*m[22] + f[23]*m[23])*m[3] + (f[24]*m[18] + f[25]*m[19] + f[26]*m[20] + f[27]*m[21] + f[28]*m[22] + f[29]*m[23])*m[4] + (f[30]*m[18] + f[31]*m[19] + f[32]*m[20] + f[33]*m[21] + f[34]*m[22] + f[35]*m[23])*m[5]) + 1.0*pow(x, 2.0)*((f[0]*m[12] + f[1]*m[13] + f[2]*m[14] + f[3]*m[15] + f[4]*m[16] + f[5]*m[17])*m[0] + (f[6]*m[12] + f[7]*m[13] + f[8]*m[14] + f[9]*m[15] + f[10]*m[16] + f[11]*m[17])*m[1] + (f[12]*m[12] + f[13]*m[13] + f[14]*m[14] + f[15]*m[15] + f[16]*m[16] + f[17]*m[17])*m[2] + (f[18]*m[12] + f[19]*m[13] + f[20]*m[14] + f[21]*m[15] + f[22]*m[16] + f[23]*m[17])*m[3] + (f[24]*m[12] + f[25]*m[13] + f[26]*m[14] + f[27]*m[15] + f[28]*m[16] + f[29]*m[17])*m[4] + (f[30]*m[12] + f[31]*m[13] + f[32]*m[14] + f[33]*m[15] + f[34]*m[16] + f[35]*m[17])*m[5]) + 1.0*x*((f[0]*m[6] + f[1]*m[7] + f[2]*m[8] + f[3]*m[9] + f[4]*m[10] + f[5]*m[11])*m[0] + (f[6]*m[6] + f[7]*m[7] + f[8]*m[8] + f[9]*m[9] + f[10]*m[10] + f[11]*m[11])*m[1] + (f[12]*m[6] + f[13]*m[7] + f[14]*m[8] + f[15]*m[9] + f[16]*m[10] + f[17]*m[11])*m[2] + (f[18]*m[6] + f[19]*m[7] + f[20]*m[8] + f[21]*m[9] + f[22]*m[10] + f[23]*m[11])*m[3] + (f[24]*m[6] + f[25]*m[7] + f[26]*m[8] + f[27]*m[9] + f[28]*m[10] + f[29]*m[11])*m[4] + (f[30]*m[6] + f[31]*m[7] + f[32]*m[8] + f[33]*m[9] + f[34]*m[10] + f[35]*m[11])*m[5]) + pow(y, 5.0)*(pow(x, 5.0)*((f[0]*m[30] + f[1]*m[31] + f[2]*m[32] + f[3]*m[33] + f[4]*m[34] + f[5]*m[35])*m[30] + (f[6]*m[30] + f[7]*m[31] + f[8]*m[32] + f[9]*m[33] + f[10]*m[34] + f[11]*m[35])*m[31] + (f[12]*m[30] + f[13]*m[31] + f[14]*m[32] + f[15]*m[33] + f[16]*m[34] + f[17]*m[35])*m[32] + (f[18]*m[30] + f[19]*m[31] + f[20]*m[32] + f[21]*m[33] + f[22]*m[34] + f[23]*m[35])*m[33] + (f[24]*m[30] + f[25]*m[31] + f[26]*m[32] + f[27]*m[33] + f[28]*m[34] + f[29]*m[35])*m[34] + (f[30]*m[30] + f[31]*m[31] + f[32]*m[32] + f[33]*m[33] + f[34]*m[34] + f[35]*m[35])*m[35]) + pow(x, 4.0)*((f[0]*m[24] + f[1]*m[25] + f[2]*m[26] + f[3]*m[27] + f[4]*m[28] + f[5]*m[29])*m[30] + (f[6]*m[24] + f[7]*m[25] + f[8]*m[26] + f[9]*m[27] + f[10]*m[28] + f[11]*m[29])*m[31] + (f[12]*m[24] + f[13]*m[25] + f[14]*m[26] + f[15]*m[27] + f[16]*m[28] + f[17]*m[29])*m[32] + (f[18]*m[24] + f[19]*m[25] + f[20]*m[26] + f[21]*m[27] + f[22]*m[28] + f[23]*m[29])*m[33] + (f[24]*m[24] + f[25]*m[25] + f[26]*m[26] + f[27]*m[27] + f[28]*m[28] + f[29]*m[29])*m[34] + (f[30]*m[24] + f[31]*m[25] + f[32]*m[26] + f[33]*m[27] + f[34]*m[28] + f[35]*m[29])*m[35]) + pow(x, 3.0)*((f[0]*m[18] + f[1]*m[19] + f[2]*m[20] + f[3]*m[21] + f[4]*m[22] + f[5]*m[23])*m[30] + (f[6]*m[18] + f[7]*m[19] + f[8]*m[20] + f[9]*m[21] + f[10]*m[22] + f[11]*m[23])*m[31] + (f[12]*m[18] + f[13]*m[19] + f[14]*m[20] + f[15]*m[21] + f[16]*m[22] + f[17]*m[23])*m[32] + (f[18]*m[18] + f[19]*m[19] + f[20]*m[20] + f[21]*m[21] + f[22]*m[22] + f[23]*m[23])*m[33] + (f[24]*m[18] + f[25]*m[19] + f[26]*m[20] + f[27]*m[21] + f[28]*m[22] + f[29]*m[23])*m[34] + (f[30]*m[18] + f[31]*m[19] + f[32]*m[20] + f[33]*m[21] + f[34]*m[22] + f[35]*m[23])*m[35]) + pow(x, 2.0)*((f[0]*m[12] + f[1]*m[13] + f[2]*m[14] + f[3]*m[15] + f[4]*m[16] + f[5]*m[17])*m[30] + (f[6]*m[12] + f[7]*m[13] + f[8]*m[14] + f[9]*m[15] + f[10]*m[16] + f[11]*m[17])*m[31] + (f[12]*m[12] + f[13]*m[13] + f[14]*m[14] + f[15]*m[15] + f[16]*m[16] + f[17]*m[17])*m[32] + (f[18]*m[12] + f[19]*m[13] + f[20]*m[14] + f[21]*m[15] + f[22]*m[16] + f[23]*m[17])*m[33] + (f[24]*m[12] + f[25]*m[13] + f[26]*m[14] + f[27]*m[15] + f[28]*m[16] + f[29]*m[17])*m[34] + (f[30]*m[12] + f[31]*m[13] + f[32]*m[14] + f[33]*m[15] + f[34]*m[16] + f[35]*m[17])*m[35]) + x*((f[0]*m[6] + f[1]*m[7] + f[2]*m[8] + f[3]*m[9] + f[4]*m[10] + f[5]*m[11])*m[30] + (f[6]*m[6] + f[7]*m[7] + f[8]*m[8] + f[9]*m[9] + f[10]*m[10] + f[11]*m[11])*m[31] + (f[12]*m[6] + f[13]*m[7] + f[14]*m[8] + f[15]*m[9] + f[16]*m[10] + f[17]*m[11])*m[32] + (f[18]*m[6] + f[19]*m[7] + f[20]*m[8] + f[21]*m[9] + f[22]*m[10] + f[23]*m[11])*m[33] + (f[24]*m[6] + f[25]*m[7] + f[26]*m[8] + f[27]*m[9] + f[28]*m[10] + f[29]*m[11])*m[34] + (f[30]*m[6] + f[31]*m[7] + f[32]*m[8] + f[33]*m[9] + f[34]*m[10] + f[35]*m[11])*m[35]) + 1.0*(f[0]*m[0] + f[1]*m[1] + f[2]*m[2] + f[3]*m[3] + f[4]*m[4] + f[5]*m[5])*m[30] + 1.0*(f[6]*m[0] + f[7]*m[1] + f[8]*m[2] + f[9]*m[3] + f[10]*m[4] + f[11]*m[5])*m[31] + 1.0*(f[12]*m[0] + f[13]*m[1] + f[14]*m[2] + f[15]*m[3] + f[16]*m[4] + f[17]*m[5])*m[32] + 1.0*(f[18]*m[0] + f[19]*m[1] + f[20]*m[2] + f[21]*m[3] + f[22]*m[4] + f[23]*m[5])*m[33] + 1.0*(f[24]*m[0] + f[25]*m[1] + f[26]*m[2] + f[27]*m[3] + f[28]*m[4] + f[29]*m[5])*m[34] + 1.0*(f[30]*m[0] + f[31]*m[1] + f[32]*m[2] + f[33]*m[3] + f[34]*m[4] + f[35]*m[5])*m[35]) + pow(y, 4.0)*(pow(x, 5.0)*((f[0]*m[30] + f[1]*m[31] + f[2]*m[32] + f[3]*m[33] + f[4]*m[34] + f[5]*m[35])*m[24] + (f[6]*m[30] + f[7]*m[31] + f[8]*m[32] + f[9]*m[33] + f[10]*m[34] + f[11]*m[35])*m[25] + (f[12]*m[30] + f[13]*m[31] + f[14]*m[32] + f[15]*m[33] + f[16]*m[34] + f[17]*m[35])*m[26] + (f[18]*m[30] + f[19]*m[31] + f[20]*m[32] + f[21]*m[33] + f[22]*m[34] + f[23]*m[35])*m[27] + (f[24]*m[30] + f[25]*m[31] + f[26]*m[32] + f[27]*m[33] + f[28]*m[34] + f[29]*m[35])*m[28] + (f[30]*m[30] + f[31]*m[31] + f[32]*m[32] + f[33]*m[33] + f[34]*m[34] + f[35]*m[35])*m[29]) + pow(x, 4.0)*((f[0]*m[24] + f[1]*m[25] + f[2]*m[26] + f[3]*m[27] + f[4]*m[28] + f[5]*m[29])*m[24] + (f[6]*m[24] + f[7]*m[25] + f[8]*m[26] + f[9]*m[27] + f[10]*m[28] + f[11]*m[29])*m[25] + (f[12]*m[24] + f[13]*m[25] + f[14]*m[26] + f[15]*m[27] + f[16]*m[28] + f[17]*m[29])*m[26] + (f[18]*m[24] + f[19]*m[25] + f[20]*m[26] + f[21]*m[27] + f[22]*m[28] + f[23]*m[29])*m[27] + (f[24]*m[24] + f[25]*m[25] + f[26]*m[26] + f[27]*m[27] + f[28]*m[28] + f[29]*m[29])*m[28] + (f[30]*m[24] + f[31]*m[25] + f[32]*m[26] + f[33]*m[27] + f[34]*m[28] + f[35]*m[29])*m[29]) + pow(x, 3.0)*((f[0]*m[18] + f[1]*m[19] + f[2]*m[20] + f[3]*m[21] + f[4]*m[22] + f[5]*m[23])*m[24] + (f[6]*m[18] + f[7]*m[19] + f[8]*m[20] + f[9]*m[21] + f[10]*m[22] + f[11]*m[23])*m[25] + (f[12]*m[18] + f[13]*m[19] + f[14]*m[20] + f[15]*m[21] + f[16]*m[22] + f[17]*m[23])*m[26] + (f[18]*m[18] + f[19]*m[19] + f[20]*m[20] + f[21]*m[21] + f[22]*m[22] + f[23]*m[23])*m[27] + (f[24]*m[18] + f[25]*m[19] + f[26]*m[20] + f[27]*m[21] + f[28]*m[22] + f[29]*m[23])*m[28] + (f[30]*m[18] + f[31]*m[19] + f[32]*m[20] + f[33]*m[21] + f[34]*m[22] + f[35]*m[23])*m[29]) + pow(x, 2.0)*((f[0]*m[12] + f[1]*m[13] + f[2]*m[14] + f[3]*m[15] + f[4]*m[16] + f[5]*m[17])*m[24] + (f[6]*m[12] + f[7]*m[13] + f[8]*m[14] + f[9]*m[15] + f[10]*m[16] + f[11]*m[17])*m[25] + (f[12]*m[12] + f[13]*m[13] + f[14]*m[14] + f[15]*m[15] + f[16]*m[16] + f[17]*m[17])*m[26] + (f[18]*m[12] + f[19]*m[13] + f[20]*m[14] + f[21]*m[15] + f[22]*m[16] + f[23]*m[17])*m[27] + (f[24]*m[12] + f[25]*m[13] + f[26]*m[14] + f[27]*m[15] + f[28]*m[16] + f[29]*m[17])*m[28] + (f[30]*m[12] + f[31]*m[13] + f[32]*m[14] + f[33]*m[15] + f[34]*m[16] + f[35]*m[17])*m[29]) + x*((f[0]*m[6] + f[1]*m[7] + f[2]*m[8] + f[3]*m[9] + f[4]*m[10] + f[5]*m[11])*m[24] + (f[6]*m[6] + f[7]*m[7] + f[8]*m[8] + f[9]*m[9] + f[10]*m[10] + f[11]*m[11])*m[25] + (f[12]*m[6] + f[13]*m[7] + f[14]*m[8] + f[15]*m[9] + f[16]*m[10] + f[17]*m[11])*m[26] + (f[18]*m[6] + f[19]*m[7] + f[20]*m[8] + f[21]*m[9] + f[22]*m[10] + f[23]*m[11])*m[27] + (f[24]*m[6] + f[25]*m[7] + f[26]*m[8] + f[27]*m[9] + f[28]*m[10] + f[29]*m[11])*m[28] + (f[30]*m[6] + f[31]*m[7] + f[32]*m[8] + f[33]*m[9] + f[34]*m[10] + f[35]*m[11])*m[29]) + 1.0*(f[0]*m[0] + f[1]*m[1] + f[2]*m[2] + f[3]*m[3] + f[4]*m[4] + f[5]*m[5])*m[24] + 1.0*(f[6]*m[0] + f[7]*m[1] + f[8]*m[2] + f[9]*m[3] + f[10]*m[4] + f[11]*m[5])*m[25] + 1.0*(f[12]*m[0] + f[13]*m[1] + f[14]*m[2] + f[15]*m[3] + f[16]*m[4] + f[17]*m[5])*m[26] + 1.0*(f[18]*m[0] + f[19]*m[1] + f[20]*m[2] + f[21]*m[3] + f[22]*m[4] + f[23]*m[5])*m[27] + 1.0*(f[24]*m[0] + f[25]*m[1] + f[26]*m[2] + f[27]*m[3] + f[28]*m[4] + f[29]*m[5])*m[28] + 1.0*(f[30]*m[0] + f[31]*m[1] + f[32]*m[2] + f[33]*m[3] + f[34]*m[4] + f[35]*m[5])*m[29]) + pow(y, 3.0)*(pow(x, 5.0)*((f[0]*m[30] + f[1]*m[31] + f[2]*m[32] + f[3]*m[33] + f[4]*m[34] + f[5]*m[35])*m[18] + (f[6]*m[30] + f[7]*m[31] + f[8]*m[32] + f[9]*m[33] + f[10]*m[34] + f[11]*m[35])*m[19] + (f[12]*m[30] + f[13]*m[31] + f[14]*m[32] + f[15]*m[33] + f[16]*m[34] + f[17]*m[35])*m[20] + (f[18]*m[30] + f[19]*m[31] + f[20]*m[32] + f[21]*m[33] + f[22]*m[34] + f[23]*m[35])*m[21] + (f[24]*m[30] + f[25]*m[31] + f[26]*m[32] + f[27]*m[33] + f[28]*m[34] + f[29]*m[35])*m[22] + (f[30]*m[30] + f[31]*m[31] + f[32]*m[32] + f[33]*m[33] + f[34]*m[34] + f[35]*m[35])*m[23]) + pow(x, 4.0)*((f[0]*m[24] + f[1]*m[25] + f[2]*m[26] + f[3]*m[27] + f[4]*m[28] + f[5]*m[29])*m[18] + (f[6]*m[24] + f[7]*m[25] + f[8]*m[26] + f[9]*m[27] + f[10]*m[28] + f[11]*m[29])*m[19] + (f[12]*m[24] + f[13]*m[25] + f[14]*m[26] + f[15]*m[27] + f[16]*m[28] + f[17]*m[29])*m[20] + (f[18]*m[24] + f[19]*m[25] + f[20]*m[26] + f[21]*m[27] + f[22]*m[28] + f[23]*m[29])*m[21] + (f[24]*m[24] + f[25]*m[25] + f[26]*m[26] + f[27]*m[27] + f[28]*m[28] + f[29]*m[29])*m[22] + (f[30]*m[24] + f[31]*m[25] + f[32]*m[26] + f[33]*m[27] + f[34]*m[28] + f[35]*m[29])*m[23]) + pow(x, 3.0)*((f[0]*m[18] + f[1]*m[19] + f[2]*m[20] + f[3]*m[21] + f[4]*m[22] + f[5]*m[23])*m[18] + (f[6]*m[18] + f[7]*m[19] + f[8]*m[20] + f[9]*m[21] + f[10]*m[22] + f[11]*m[23])*m[19] + (f[12]*m[18] + f[13]*m[19] + f[14]*m[20] + f[15]*m[21] + f[16]*m[22] + f[17]*m[23])*m[20] + (f[18]*m[18] + f[19]*m[19] + f[20]*m[20] + f[21]*m[21] + f[22]*m[22] + f[23]*m[23])*m[21] + (f[24]*m[18] + f[25]*m[19] + f[26]*m[20] + f[27]*m[21] + f[28]*m[22] + f[29]*m[23])*m[22] + (f[30]*m[18] + f[31]*m[19] + f[32]*m[20] + f[33]*m[21] + f[34]*m[22] + f[35]*m[23])*m[23]) + pow(x, 2.0)*((f[0]*m[12] + f[1]*m[13] + f[2]*m[14] + f[3]*m[15] + f[4]*m[16] + f[5]*m[17])*m[18] + (f[6]*m[12] + f[7]*m[13] + f[8]*m[14] + f[9]*m[15] + f[10]*m[16] + f[11]*m[17])*m[19] + (f[12]*m[12] + f[13]*m[13] + f[14]*m[14] + f[15]*m[15] + f[16]*m[16] + f[17]*m[17])*m[20] + (f[18]*m[12] + f[19]*m[13] + f[20]*m[14] + f[21]*m[15] + f[22]*m[16] + f[23]*m[17])*m[21] + (f[24]*m[12] + f[25]*m[13] + f[26]*m[14] + f[27]*m[15] + f[28]*m[16] + f[29]*m[17])*m[22] + (f[30]*m[12] + f[31]*m[13] + f[32]*m[14] + f[33]*m[15] + f[34]*m[16] + f[35]*m[17])*m[23]) + x*((f[0]*m[6] + f[1]*m[7] + f[2]*m[8] + f[3]*m[9] + f[4]*m[10] + f[5]*m[11])*m[18] + (f[6]*m[6] + f[7]*m[7] + f[8]*m[8] + f[9]*m[9] + f[10]*m[10] + f[11]*m[11])*m[19] + (f[12]*m[6] + f[13]*m[7] + f[14]*m[8] + f[15]*m[9] + f[16]*m[10] + f[17]*m[11])*m[20] + (f[18]*m[6] + f[19]*m[7] + f[20]*m[8] + f[21]*m[9] + f[22]*m[10] + f[23]*m[11])*m[21] + (f[24]*m[6] + f[25]*m[7] + f[26]*m[8] + f[27]*m[9] + f[28]*m[10] + f[29]*m[11])*m[22] + (f[30]*m[6] + f[31]*m[7] + f[32]*m[8] + f[33]*m[9] + f[34]*m[10] + f[35]*m[11])*m[23]) + 1.0*(f[0]*m[0] + f[1]*m[1] + f[2]*m[2] + f[3]*m[3] + f[4]*m[4] + f[5]*m[5])*m[18] + 1.0*(f[6]*m[0] + f[7]*m[1] + f[8]*m[2] + f[9]*m[3] + f[10]*m[4] + f[11]*m[5])*m[19] + 1.0*(f[12]*m[0] + f[13]*m[1] + f[14]*m[2] + f[15]*m[3] + f[16]*m[4] + f[17]*m[5])*m[20] + 1.0*(f[18]*m[0] + f[19]*m[1] + f[20]*m[2] + f[21]*m[3] + f[22]*m[4] + f[23]*m[5])*m[21] + 1.0*(f[24]*m[0] + f[25]*m[1] + f[26]*m[2] + f[27]*m[3] + f[28]*m[4] + f[29]*m[5])*m[22] + 1.0*(f[30]*m[0] + f[31]*m[1] + f[32]*m[2] + f[33]*m[3] + f[34]*m[4] + f[35]*m[5])*m[23]) + pow(y, 2.0)*(pow(x, 5.0)*((f[0]*m[30] + f[1]*m[31] + f[2]*m[32] + f[3]*m[33] + f[4]*m[34] + f[5]*m[35])*m[12] + (f[6]*m[30] + f[7]*m[31] + f[8]*m[32] + f[9]*m[33] + f[10]*m[34] + f[11]*m[35])*m[13] + (f[12]*m[30] + f[13]*m[31] + f[14]*m[32] + f[15]*m[33] + f[16]*m[34] + f[17]*m[35])*m[14] + (f[18]*m[30] + f[19]*m[31] + f[20]*m[32] + f[21]*m[33] + f[22]*m[34] + f[23]*m[35])*m[15] + (f[24]*m[30] + f[25]*m[31] + f[26]*m[32] + f[27]*m[33] + f[28]*m[34] + f[29]*m[35])*m[16] + (f[30]*m[30] + f[31]*m[31] + f[32]*m[32] + f[33]*m[33] + f[34]*m[34] + f[35]*m[35])*m[17]) + pow(x, 4.0)*((f[0]*m[24] + f[1]*m[25] + f[2]*m[26] + f[3]*m[27] + f[4]*m[28] + f[5]*m[29])*m[12] + (f[6]*m[24] + f[7]*m[25] + f[8]*m[26] + f[9]*m[27] + f[10]*m[28] + f[11]*m[29])*m[13] + (f[12]*m[24] + f[13]*m[25] + f[14]*m[26] + f[15]*m[27] + f[16]*m[28] + f[17]*m[29])*m[14] + (f[18]*m[24] + f[19]*m[25] + f[20]*m[26] + f[21]*m[27] + f[22]*m[28] + f[23]*m[29])*m[15] + (f[24]*m[24] + f[25]*m[25] + f[26]*m[26] + f[27]*m[27] + f[28]*m[28] + f[29]*m[29])*m[16] + (f[30]*m[24] + f[31]*m[25] + f[32]*m[26] + f[33]*m[27] + f[34]*m[28] + f[35]*m[29])*m[17]) + pow(x, 3.0)*((f[0]*m[18] + f[1]*m[19] + f[2]*m[20] + f[3]*m[21] + f[4]*m[22] + f[5]*m[23])*m[12] + (f[6]*m[18] + f[7]*m[19] + f[8]*m[20] + f[9]*m[21] + f[10]*m[22] + f[11]*m[23])*m[13] + (f[12]*m[18] + f[13]*m[19] + f[14]*m[20] + f[15]*m[21] + f[16]*m[22] + f[17]*m[23])*m[14] + (f[18]*m[18] + f[19]*m[19] + f[20]*m[20] + f[21]*m[21] + f[22]*m[22] + f[23]*m[23])*m[15] + (f[24]*m[18] + f[25]*m[19] + f[26]*m[20] + f[27]*m[21] + f[28]*m[22] + f[29]*m[23])*m[16] + (f[30]*m[18] + f[31]*m[19] + f[32]*m[20] + f[33]*m[21] + f[34]*m[22] + f[35]*m[23])*m[17]) + pow(x, 2.0)*((f[0]*m[12] + f[1]*m[13] + f[2]*m[14] + f[3]*m[15] + f[4]*m[16] + f[5]*m[17])*m[12] + (f[6]*m[12] + f[7]*m[13] + f[8]*m[14] + f[9]*m[15] + f[10]*m[16] + f[11]*m[17])*m[13] + (f[12]*m[12] + f[13]*m[13] + f[14]*m[14] + f[15]*m[15] + f[16]*m[16] + f[17]*m[17])*m[14] + (f[18]*m[12] + f[19]*m[13] + f[20]*m[14] + f[21]*m[15] + f[22]*m[16] + f[23]*m[17])*m[15] + (f[24]*m[12] + f[25]*m[13] + f[26]*m[14] + f[27]*m[15] + f[28]*m[16] + f[29]*m[17])*m[16] + (f[30]*m[12] + f[31]*m[13] + f[32]*m[14] + f[33]*m[15] + f[34]*m[16] + f[35]*m[17])*m[17]) + x*((f[0]*m[6] + f[1]*m[7] + f[2]*m[8] + f[3]*m[9] + f[4]*m[10] + f[5]*m[11])*m[12] + (f[6]*m[6] + f[7]*m[7] + f[8]*m[8] + f[9]*m[9] + f[10]*m[10] + f[11]*m[11])*m[13] + (f[12]*m[6] + f[13]*m[7] + f[14]*m[8] + f[15]*m[9] + f[16]*m[10] + f[17]*m[11])*m[14] + (f[18]*m[6] + f[19]*m[7] + f[20]*m[8] + f[21]*m[9] + f[22]*m[10] + f[23]*m[11])*m[15] + (f[24]*m[6] + f[25]*m[7] + f[26]*m[8] + f[27]*m[9] + f[28]*m[10] + f[29]*m[11])*m[16] + (f[30]*m[6] + f[31]*m[7] + f[32]*m[8] + f[33]*m[9] + f[34]*m[10] + f[35]*m[11])*m[17]) + 1.0*(f[0]*m[0] + f[1]*m[1] + f[2]*m[2] + f[3]*m[3] + f[4]*m[4] + f[5]*m[5])*m[12] + 1.0*(f[6]*m[0] + f[7]*m[1] + f[8]*m[2] + f[9]*m[3] + f[10]*m[4] + f[11]*m[5])*m[13] + 1.0*(f[12]*m[0] + f[13]*m[1] + f[14]*m[2] + f[15]*m[3] + f[16]*m[4] + f[17]*m[5])*m[14] + 1.0*(f[18]*m[0] + f[19]*m[1] + f[20]*m[2] + f[21]*m[3] + f[22]*m[4] + f[23]*m[5])*m[15] + 1.0*(f[24]*m[0] + f[25]*m[1] + f[26]*m[2] + f[27]*m[3] + f[28]*m[4] + f[29]*m[5])*m[16] + 1.0*(f[30]*m[0] + f[31]*m[1] + f[32]*m[2] + f[33]*m[3] + f[34]*m[4] + f[35]*m[5])*m[17]) + y*(pow(x, 5.0)*((f[0]*m[30] + f[1]*m[31] + f[2]*m[32] + f[3]*m[33] + f[4]*m[34] + f[5]*m[35])*m[6] + (f[6]*m[30] + f[7]*m[31] + f[8]*m[32] + f[9]*m[33] + f[10]*m[34] + f[11]*m[35])*m[7] + (f[12]*m[30] + f[13]*m[31] + f[14]*m[32] + f[15]*m[33] + f[16]*m[34] + f[17]*m[35])*m[8] + (f[18]*m[30] + f[19]*m[31] + f[20]*m[32] + f[21]*m[33] + f[22]*m[34] + f[23]*m[35])*m[9] + (f[24]*m[30] + f[25]*m[31] + f[26]*m[32] + f[27]*m[33] + f[28]*m[34] + f[29]*m[35])*m[10] + (f[30]*m[30] + f[31]*m[31] + f[32]*m[32] + f[33]*m[33] + f[34]*m[34] + f[35]*m[35])*m[11]) + pow(x, 4.0)*((f[0]*m[24] + f[1]*m[25] + f[2]*m[26] + f[3]*m[27] + f[4]*m[28] + f[5]*m[29])*m[6] + (f[6]*m[24] + f[7]*m[25] + f[8]*m[26] + f[9]*m[27] + f[10]*m[28] + f[11]*m[29])*m[7] + (f[12]*m[24] + f[13]*m[25] + f[14]*m[26] + f[15]*m[27] + f[16]*m[28] + f[17]*m[29])*m[8] + (f[18]*m[24] + f[19]*m[25] + f[20]*m[26] + f[21]*m[27] + f[22]*m[28] + f[23]*m[29])*m[9] + (f[24]*m[24] + f[25]*m[25] + f[26]*m[26] + f[27]*m[27] + f[28]*m[28] + f[29]*m[29])*m[10] + (f[30]*m[24] + f[31]*m[25] + f[32]*m[26] + f[33]*m[27] + f[34]*m[28] + f[35]*m[29])*m[11]) + pow(x, 3.0)*((f[0]*m[18] + f[1]*m[19] + f[2]*m[20] + f[3]*m[21] + f[4]*m[22] + f[5]*m[23])*m[6] + (f[6]*m[18] + f[7]*m[19] + f[8]*m[20] + f[9]*m[21] + f[10]*m[22] + f[11]*m[23])*m[7] + (f[12]*m[18] + f[13]*m[19] + f[14]*m[20] + f[15]*m[21] + f[16]*m[22] + f[17]*m[23])*m[8] + (f[18]*m[18] + f[19]*m[19] + f[20]*m[20] + f[21]*m[21] + f[22]*m[22] + f[23]*m[23])*m[9] + (f[24]*m[18] + f[25]*m[19] + f[26]*m[20] + f[27]*m[21] + f[28]*m[22] + f[29]*m[23])*m[10] + (f[30]*m[18] + f[31]*m[19] + f[32]*m[20] + f[33]*m[21] + f[34]*m[22] + f[35]*m[23])*m[11]) + pow(x, 2.0)*((f[0]*m[12] + f[1]*m[13] + f[2]*m[14] + f[3]*m[15] + f[4]*m[16] + f[5]*m[17])*m[6] + (f[6]*m[12] + f[7]*m[13] + f[8]*m[14] + f[9]*m[15] + f[10]*m[16] + f[11]*m[17])*m[7] + (f[12]*m[12] + f[13]*m[13] + f[14]*m[14] + f[15]*m[15] + f[16]*m[16] + f[17]*m[17])*m[8] + (f[18]*m[12] + f[19]*m[13] + f[20]*m[14] + f[21]*m[15] + f[22]*m[16] + f[23]*m[17])*m[9] + (f[24]*m[12] + f[25]*m[13] + f[26]*m[14] + f[27]*m[15] + f[28]*m[16] + f[29]*m[17])*m[10] + (f[30]*m[12] + f[31]*m[13] + f[32]*m[14] + f[33]*m[15] + f[34]*m[16] + f[35]*m[17])*m[11]) + x*((f[0]*m[6] + f[1]*m[7] + f[2]*m[8] + f[3]*m[9] + f[4]*m[10] + f[5]*m[11])*m[6] + (f[6]*m[6] + f[7]*m[7] + f[8]*m[8] + f[9]*m[9] + f[10]*m[10] + f[11]*m[11])*m[7] + (f[12]*m[6] + f[13]*m[7] + f[14]*m[8] + f[15]*m[9] + f[16]*m[10] + f[17]*m[11])*m[8] + (f[18]*m[6] + f[19]*m[7] + f[20]*m[8] + f[21]*m[9] + f[22]*m[10] + f[23]*m[11])*m[9] + (f[24]*m[6] + f[25]*m[7] + f[26]*m[8] + f[27]*m[9] + f[28]*m[10] + f[29]*m[11])*m[10] + (f[30]*m[6] + f[31]*m[7] + f[32]*m[8] + f[33]*m[9] + f[34]*m[10] + f[35]*m[11])*m[11]) + 1.0*(f[0]*m[0] + f[1]*m[1] + f[2]*m[2] + f[3]*m[3] + f[4]*m[4] + f[5]*m[5])*m[6] + 1.0*(f[6]*m[0] + f[7]*m[1] + f[8]*m[2] + f[9]*m[3] + f[10]*m[4] + f[11]*m[5])*m[7] + 1.0*(f[12]*m[0] + f[13]*m[1] + f[14]*m[2] + f[15]*m[3] + f[16]*m[4] + f[17]*m[5])*m[8] + 1.0*(f[18]*m[0] + f[19]*m[1] + f[20]*m[2] + f[21]*m[3] + f[22]*m[4] + f[23]*m[5])*m[9] + 1.0*(f[24]*m[0] + f[25]*m[1] + f[26]*m[2] + f[27]*m[3] + f[28]*m[4] + f[29]*m[5])*m[10] + 1.0*(f[30]*m[0] + f[31]*m[1] + f[32]*m[2] + f[33]*m[3] + f[34]*m[4] + f[35]*m[5])*m[11]) + 1.0*(f[0]*m[0] + f[1]*m[1] + f[2]*m[2] + f[3]*m[3] + f[4]*m[4] + f[5]*m[5])*m[0] + 1.0*(f[6]*m[0] + f[7]*m[1] + f[8]*m[2] + f[9]*m[3] + f[10]*m[4] + f[11]*m[5])*m[1] + 1.0*(f[12]*m[0] + f[13]*m[1] + f[14]*m[2] + f[15]*m[3] + f[16]*m[4] + f[17]*m[5])*m[2] + 1.0*(f[18]*m[0] + f[19]*m[1] + f[20]*m[2] + f[21]*m[3] + f[22]*m[4] + f[23]*m[5])*m[3] + 1.0*(f[24]*m[0] + f[25]*m[1] + f[26]*m[2] + f[27]*m[3] + f[28]*m[4] + f[29]*m[5])*m[4] + 1.0*(f[30]*m[0] + f[31]*m[1] + f[32]*m[2] + f[33]*m[3] + f[34]*m[4] + f[35]*m[5])*m[5];
}