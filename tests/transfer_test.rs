use anyhow::Result;
use cuda_texture_transfer::{CudaBuffer, Size, TextureReceiver, TextureSender};
use glium::{
    backend::Facade,
    texture::{ClientFormat, MipmapsOption, RawImage2d, UncompressedFloatFormat},
    GlObject, Texture2d,
};
use std::borrow::Cow;
use tonari_gl::HeadlessGlContext;

fn main() -> Result<()> {
    let size = Size {
        width: 640,
        height: 480,
    };

    let (context, event_loop) = HeadlessGlContext::new_root_context::<()>(size).unwrap();
    let context = context.into_not_current();

    let headless = context
        .create_headless_shared_context_for_textures_only(&event_loop)
        .into_current()
        .unwrap()
        .into_glium_context();

    let original_texture_data = vec![128u8; size.num_pixels() * 4];

    let gl_texture = create_texture(&headless, &original_texture_data, size)?;

    let mut cuda_buffer = CudaBuffer::new(size)?;

    let mut texture_sender = TextureSender::new();
    texture_sender.copy_texture_to_cuda_buffer(gl_texture.get_id(), size, &mut cuda_buffer)?;

    let thread_gl_context = context.create_headless_shared_context_for_textures_only(&event_loop);

    let thread_handle = std::thread::spawn(move || -> Result<_> {
        let thread_gl_context = thread_gl_context.into_current().unwrap();
        let headless_thread_gl_context = thread_gl_context.into_glium_context();

        let black_texture_data = vec![0u8; size.num_pixels() * 4];
        let thread_gl_texture =
            create_texture(&headless_thread_gl_context, &black_texture_data, size).unwrap();

        let mut texture_receiver = TextureReceiver::new();
        texture_receiver.copy_cuda_buffer_to_texture(
            &cuda_buffer,
            thread_gl_texture.get_id(),
            size,
        )?;

        let texture_data: Vec<Vec<(u8, u8, u8, u8)>> = thread_gl_texture.read();
        Ok(texture_data)
    });

    let result = thread_handle.join().unwrap()?;

    let transferred_data: Vec<u8> = result
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
