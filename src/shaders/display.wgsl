
@group(0) @binding(0)
var source_img : texture_2d<f32>;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
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
    let tex_size = vec2<f32>(textureDimensions(source_img));
    let pixel_pos = vec2<i32>(tex_size * vertex_in.tex_coord);
    let color =textureLoad(source_img, pixel_pos,0);

    return clamp(color, vec4<f32>(0.), vec4<f32>(1.0));
}
