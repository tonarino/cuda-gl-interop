use anyhow::Result;
use cuda_gl_interop::{CudaBuffer, Size, TextureReceiver, TextureSender};
use glium::{
    backend::{glutin::SimpleWindowBuilder, Facade},
    texture::{ClientFormat, MipmapsOption, RawImage2d, UncompressedFloatFormat},
    GlObject, Texture2d,
};
use std::borrow::Cow;
use winit::event_loop::EventLoop;

fn main() -> Result<()> {
    let size = Size {
        width: 640,
        height: 480,
    };

    let event_loop = EventLoop::builder().build().unwrap();
    let (_window, display) = SimpleWindowBuilder::new().build(&event_loop);

    let original_texture_data = vec![128u8; size.num_pixels() * 4];
    let original_texture = create_texture(&display, &original_texture_data, size)?;

    let mut cuda_buffer = CudaBuffer::new(size)?;

    let mut texture_sender = TextureSender::new();
    texture_sender.copy_texture_to_cuda_buffer(
        original_texture.get_id(),
        size,
        &mut cuda_buffer,
    )?;

    let black_texture_data = vec![0u8; size.num_pixels() * 4];
    let copied_texture = create_texture(&display, &black_texture_data, size).unwrap();

    let mut texture_receiver = TextureReceiver::new();
    texture_receiver.copy_cuda_buffer_to_texture(&cuda_buffer, copied_texture.get_id(), size)?;

    let copied_texture_data: Vec<Vec<(u8, u8, u8, u8)>> = copied_texture.read();

    let transferred_data: Vec<u8> = copied_texture_data
        .into_iter()
        .flat_map(|pixel| pixel.into_iter().flat_map(|(r, g, b, a)| [r, g, b, a]))
        .collect();

    assert_eq!(original_texture_data, transferred_data);

    println!("CUDA texture transfer test passed.");
    Ok(())
}

fn create_texture<F: Facade>(facade: &F, data: &[u8], size: Size) -> Result<Texture2d> {
    let rgba_data = RawImage2d {
        data: Cow::Borrowed(data),
        width: size.width,
        height: size.height,
        format: ClientFormat::U8U8U8U8,
    };

    Ok(Texture2d::with_format(
        facade,
        rgba_data,
        UncompressedFloatFormat::U8U8U8U8,
        MipmapsOption::NoMipmap,
    )?)
}
