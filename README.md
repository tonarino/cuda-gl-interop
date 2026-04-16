# cuda-gl-interop

Crate for copying memory between CUDA buffers and OpenGL textures.

## Example

## CUDA buffer to OpenGL texture

```rust
    let mut receiver = TextureReceiver::new();

    let size = Size::new(640, 480);
    let buffer = CudaBuffer::new(size).expect("CUDA buffer can be allocated");
    // ...fill the buffer with data.

    let texture_id = 0;
    // ...create a 640x480 RGBA8 texture.

    receiver
        .copy_cuda_buffer_to_texture(&buffer, texture_id, size)
        .expect("CUDA buffer can be copied to a texture");
```

## OpenGL texture to CUDA buffer

```rust
    let mut sender = TextureSender::new();

    let texture_id = 0;
    // ...create a 640x480 RGBA8 texture and fill it with data.

    let size = Size::new(640, 480);
    let buffer = CudaBuffer::new(size).expect("CUDA buffer can be allocated");

    sender
        .copy_texture_to_cuda_buffer(texture_id, size, &mut buffer)
        .expect("Texture can be copied to a CUDA buffer");
```

## Dependencies
- cargo
- rustc

## Build

```
$ cargo build --release
```

## Testing

```
$ cargo test
```

## Code Format

The formatting options currently use nightly-only options.

```
$ cargo +nightly fmt
```

## Code Linting

```
$ cargo clippy
```