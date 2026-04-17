use anyhow::{anyhow, bail, Result};
use cudarc::runtime::sys::{self as cudart, cudaMemcpyKind};
use euclid::default::Size2D;
use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::c_void,
    marker::PhantomData,
    mem::MaybeUninit,
};

const CUDA_GRAPHICS_REGISTER_FLAGS_NONE: u32 = 0;
const CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY: u32 = 1;
const CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: u32 = 2;

const BYTES_PER_RGBA8_PIXEL: usize = 4;

type CudaBufferPtr = *mut c_void;

pub type Size = Size2D<u32>;

pub struct RegisteredTexture {
    /// Invariant: `graphics_resource` is a valid and initialized FFI handle.
    graphics_resource: cudart::cudaGraphicsResource_t,
    size: Size,
    usage: TextureUsage,
}

pub struct TextureRegistry {
    /// Maps OpenGL texture ID to corresponding CUDA graphics resource FFI handle.
    texture_map: HashMap<u32, RegisteredTexture>,
}

impl Default for TextureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureRegistry {
    pub fn new() -> Self {
        Self {
            texture_map: HashMap::new(),
        }
    }

    /// # Errors
    ///
    /// Will return `Err` if `texture_id` has been previously registered, but with different
    /// `width``, `height` or `usage`.
    pub fn get_or_insert_registered_texture(
        &mut self,
        texture_id: u32,
        size: impl Into<Size>,
        usage: TextureUsage,
    ) -> Result<&mut RegisteredTexture> {
        let size = size.into();
        let entry = self.texture_map.entry(texture_id);

        let vacant_entry = match entry {
            Entry::Occupied(occupied) => {
                let texture = occupied.into_mut();
                if size != texture.size {
                    bail!(
                        "Texture id {texture_id} already registered with size {:?}, requesting it \
                         again with different size {size:?} is not supported.",
                        texture.size,
                    );
                }
                if texture.usage != usage {
                    bail!(
                        "Texture id {texture_id} already registered with usage {:?}, requesting \
                         it again with different usage {usage:?} is not supported.",
                        texture.usage,
                    );
                }
                return Ok(texture);
            }
            Entry::Vacant(vacant) => vacant,
        };

        let flags = match usage {
            TextureUsage::Read => CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
            TextureUsage::Write => CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            TextureUsage::ReadWrite => CUDA_GRAPHICS_REGISTER_FLAGS_NONE,
        };

        // SAFETY: thanks to MaybeUninit and error handling, we only use the out-pointer if the
        // FFI method has succeeded.
        let graphics_resource = unsafe {
            let mut graphics_resource_uninit =
                MaybeUninit::<cudart::cudaGraphicsResource_t>::uninit();

            cudaGraphicsGLRegisterImage(
                graphics_resource_uninit.as_mut_ptr(),
                texture_id,
                gl::TEXTURE_2D,
                flags,
            )
            .to_result()?;
            graphics_resource_uninit.assume_init()
        };

        let texture = RegisteredTexture {
            graphics_resource,
            size,
            usage,
        };

        Ok(vacant_entry.insert(texture))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TextureUsage {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug)]
pub struct CudaBuffer {
    /// Invariant: `buffer` is a valid device pointer allocated using `cudaMalloc*()` method.
    buffer: CudaBufferPtr,
    pitch: usize,
    size: Size,
}

/// SAFETY: the type represents a cudaMalloc*() allocation of device memory. The CUDA layer
/// serializes access to it by the GPU (and Rust memory safety only relates to host memory anyway).
unsafe impl Send for CudaBuffer {}

impl CudaBuffer {
    /// Allocates a GPU device buffer for an RGBA8 texture with pixel size `width`, `height`.
    pub fn new(size: impl Into<Size>) -> Result<Self> {
        let size = size.into();

        let mut pitch = 0;

        // SAFETY: thanks to MaybeUninit and error handling, we only use the out-pointer and `pitch`
        // if the FFI method has succeeded.
        let buffer = unsafe {
            let mut buffer_uninit = MaybeUninit::<CudaBufferPtr>::uninit();

            cudart::cudaMallocPitch(
                buffer_uninit.as_mut_ptr(),
                &mut pitch,
                size.width as usize * BYTES_PER_RGBA8_PIXEL,
                size.height as usize,
            )
            .to_result()?;

            buffer_uninit.assume_init()
        };

        Ok(Self {
            buffer,
            pitch,
            size,
        })
    }

    pub fn as_slice<'a>(&'a self) -> CudaSlice<'a> {
        // SAFETY: `self` is a CUDA device buffer valid for lifetime `'a`.
        unsafe { CudaSlice::new(self.buffer, self.pitch, self.size) }
    }

    pub fn as_slice_mut<'a>(&'a mut self) -> CudaSliceMut<'a> {
        // SAFETY: `self` is a CUDA device buffer valid for lifetime `'a`.
        unsafe { CudaSliceMut::new(self.buffer, self.pitch, self.size) }
    }

    pub fn ptr(&self) -> CudaBufferPtr {
        self.buffer
    }

    pub fn pitch(&self) -> usize {
        self.pitch
    }

    pub fn size(&self) -> Size {
        self.size
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // SAFETY: the struct invariant guarantees `buffer` has been appropriately allocated before.
        unsafe {
            // There is no cudaFreePitch() - cudaFree() is documented to also work with pitched
            // allocations.
            cudart::cudaFree(self.buffer)
                .to_result()
                .expect("freeing a valid CUDA pointer should not fail");
        }
    }
}

/// A slice of a CUDA device buffer.
pub struct CudaSlice<'a> {
    buffer: CudaBufferPtr,
    pitch: usize,
    size: Size,
    _phantom_data: PhantomData<&'a CudaBuffer>,
}

impl<'a> CudaSlice<'a> {
    /// # Safety
    ///
    /// The input data must represent a valid CUDA buffer, such as the one obtained
    /// with `CudaBuffer::new()`, and it must remain valid for the lifetime of `'a`.
    pub unsafe fn new(buffer: CudaBufferPtr, pitch: usize, size: impl Into<Size>) -> Self {
        Self {
            buffer,
            pitch,
            size: size.into(),
            _phantom_data: PhantomData,
        }
    }

    pub fn size(&self) -> Size {
        self.size
    }
}

/// A slice of a CUDA device buffer.
pub struct CudaSliceMut<'a> {
    buffer: CudaBufferPtr,
    pitch: usize,
    size: Size,
    _phantom_data: PhantomData<&'a mut CudaBuffer>,
}

impl<'a> CudaSliceMut<'a> {
    /// # Safety
    ///
    /// The input data must represent a valid CUDA buffer, such as the one obtained
    /// with `CudaBuffer::new()`, and it must remain valid for the lifetime of `'a`.
    pub unsafe fn new(buffer: CudaBufferPtr, pitch: usize, size: impl Into<Size>) -> Self {
        Self {
            buffer,
            pitch,
            size: size.into(),
            _phantom_data: PhantomData,
        }
    }

    pub fn size(&self) -> Size {
        self.size
    }
}

/// Utility for moving the contents of an OpenGL texture into a CUDA buffer.
/// You can then use [`TextureReceiver`] to copy the contents of the CUDA
/// buffer into an OpenGL texture on another thread.
pub struct TextureSender {
    texture_registry: TextureRegistry,
}

impl Default for TextureSender {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureSender {
    pub fn new() -> Self {
        Self {
            texture_registry: TextureRegistry::new(),
        }
    }

    /// Copies an OpenGL texture, which must be in RGBA8 format, to a CUDA buffer.
    ///
    /// This method is asynchronous (does not block the CPU until the transfer is done) from host
    /// PoV, it provides guarantees that any OpenGL operations on the texture submitted _before_
    /// this call finish before the copy, and all OpenGL operations submitted _after_ only start
    /// after the copy is finished.
    ///
    /// # Errors
    ///
    /// Will return `Err` if `cuda_buffer` was allocated with different `width` or `height`.
    /// Should also fail if `texture_id` does not identify an appropriate texture, but no guarantee.
    pub fn copy_texture_to_cuda_buffer(
        &mut self,
        texture_id: u32,
        size: impl Into<Size>,
        cuda_buffer: &mut CudaBuffer,
    ) -> Result<()> {
        self.copy_texture_to_cuda_slice(texture_id, size, cuda_buffer.as_slice_mut())
    }

    /// Copies an OpenGL texture, which must be in RGBA8 format, to a CUDA slice.
    ///
    /// This method is asynchronous (does not block the CPU until the transfer is done) from host
    /// PoV, it provides guarantees that any OpenGL operations on the texture submitted _before_
    /// this call finish before the copy, and all OpenGL operations submitted _after_ only start
    /// after the copy is finished.
    ///
    /// # Errors
    ///
    /// Will return `Err` if `cuda_slice` was allocated with different `width` or `height`.
    /// Should also fail if `texture_id` does not identify an appropriate texture, but no guarantee.
    pub fn copy_texture_to_cuda_slice(
        &mut self,
        texture_id: u32,
        size: impl Into<Size>,
        cuda_slice: CudaSliceMut<'_>,
    ) -> Result<()> {
        let size = size.into();
        let slice_size = cuda_slice.size();

        if size != slice_size {
            bail!("Passed size {size:?} differs from cuda_slice size {slice_size:?}.");
        }

        let registered_texture = self.texture_registry.get_or_insert_registered_texture(
            texture_id,
            size,
            TextureUsage::Read,
        )?;

        // CPU and GPU synchronization (fences): while cudaMemcpy2DFromArray() is documented to do
        // no host-side synchronization for device-to-device transfers [1],
        // cudaGraphicsMapResources() docs [2] say
        // > This function provides the synchronization guarantee that any graphics calls issued
        // > before cudaGraphicsMapResources() will complete before any subsequent CUDA work issued
        // > in stream begins.
        //
        // Equivalently cudaGraphicsUnmapResources() documents
        // > This function provides the synchronization guarantee that any CUDA work issued in
        // > stream before cudaGraphicsUnmapResources() will complete before any subsequently issued
        // > graphics work begins.
        //
        // The CUDA calls themselves seem to use one program-global synchronized stream per [3].
        //
        // [1] https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
        // [2] https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
        // [3] https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html

        // SAFETY: `registered_texture` was properly registered before, and we handle errors.
        unsafe {
            cudart::cudaGraphicsMapResources(
                1,
                &mut registered_texture.graphics_resource,
                std::ptr::null_mut(),
            )
            .to_result()?;
        }

        // SAFETY: the cudaGraphicsSubResourceGetMappedArray() is documented to check validity of
        // all passed parameters, we handle errors and only use the `array` out-pointer on success.
        let array = unsafe {
            let mut array_uninit = MaybeUninit::<cudart::cudaArray_t>::uninit();
            let array_index = 0;
            let mip_level = 0;

            cudart::cudaGraphicsSubResourceGetMappedArray(
                array_uninit.as_mut_ptr(),
                registered_texture.graphics_resource,
                array_index,
                mip_level,
            )
            .to_result()?;

            array_uninit.assume_init()
        };

        // SAFETY: We've checked validity of all parameters. The transfer is device-to-device, which
        // should not interact with host memory (thus Rust memory safety) anyway.
        unsafe {
            cudart::cudaMemcpy2DFromArray(
                cuda_slice.buffer,                           // dst
                cuda_slice.pitch,                            // dst pitch
                array,                                       // src
                0,                                           // width offset
                0,                                           // height offset
                size.width as usize * BYTES_PER_RGBA8_PIXEL, // width in bytes
                size.height as usize,                        // height in rows
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,    // memcpy kind
            )
            .to_result()?;
        }

        // SAFETY: this is symmetrical to cudaGraphicsMapResources() call above.
        unsafe {
            cudart::cudaGraphicsUnmapResources(
                1,
                &mut registered_texture.graphics_resource,
                std::ptr::null_mut(),
            )
            .to_result()?;
        }

        Ok(())
    }
}

/// Utility for moving the contents of a CUDA buffer into an OpenGL texture.
/// Use [`TextureSender`] to copy an OpenGL texture to a CUDA buffer, then
/// use this struct to copy contents back to another OpenGL texture.
pub struct TextureReceiver {
    texture_registry: TextureRegistry,
}

impl Default for TextureReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureReceiver {
    pub fn new() -> Self {
        Self {
            texture_registry: TextureRegistry::new(),
        }
    }

    /// Copies a CUDA buffer to an OpenGL texture, which must be in RGBA8 format.
    ///
    /// This method is asynchronous (does not block the CPU until the transfer is done) from host
    /// PoV, it provides guarantees that any OpenGL operations on the texture submitted _before_
    /// this call finish before the copy, and all OpenGL operations submitted _after_ only start
    /// after the copy is finished.
    ///
    /// # Errors
    ///
    /// Will return `Err` if `cuda_buffer` was allocated with different `width` or `height`.
    /// Should also fail if `texture_id` does not identify an appropriate texture, but no guarantee.
    pub fn copy_cuda_buffer_to_texture(
        &mut self,
        cuda_buffer: &CudaBuffer,
        texture_id: u32,
        size: impl Into<Size>,
    ) -> Result<()> {
        self.copy_cuda_slice_to_texture(cuda_buffer.as_slice(), texture_id, size)
    }

    /// Copies a CUDA slice to an OpenGL texture, which must be in RGBA8 format.
    ///
    /// This method is asynchronous (does not block the CPU until the transfer is done) from host
    /// PoV, it provides guarantees that any OpenGL operations on the texture submitted _before_
    /// this call finish before the copy, and all OpenGL operations submitted _after_ only start
    /// after the copy is finished.
    ///
    /// # Errors
    ///
    /// Will return `Err` if `cuda_slice` was allocated with different `width` or `height`.
    /// Should also fail if `texture_id` does not identify an appropriate texture, but no guarantee.
    pub fn copy_cuda_slice_to_texture(
        &mut self,
        cuda_slice: CudaSlice<'_>,
        texture_id: u32,
        size: impl Into<Size>,
    ) -> Result<()> {
        let size = size.into();
        let slice_size = cuda_slice.size();

        if size != slice_size {
            bail!("Passed size {size:?} differs from cuda_slice size {slice_size:?}.");
        }

        let registered_texture = self.texture_registry.get_or_insert_registered_texture(
            texture_id,
            size,
            TextureUsage::Write,
        )?;

        // SAFETY: `registered_texture` was properly registered before, and we handle errors.
        unsafe {
            cudart::cudaGraphicsMapResources(
                1,
                &mut registered_texture.graphics_resource,
                std::ptr::null_mut(),
            )
            .to_result()?;
        }

        // SAFETY: the cudaGraphicsSubResourceGetMappedArray() is documented to check validity of
        // all passed parameters, we handle errors and only use the `array` out-pointer on success.
        let array = unsafe {
            let mut array_uninit = MaybeUninit::<cudart::cudaArray_t>::uninit();
            let array_index = 0;
            let mip_level = 0;

            cudart::cudaGraphicsSubResourceGetMappedArray(
                array_uninit.as_mut_ptr(),
                registered_texture.graphics_resource,
                array_index,
                mip_level,
            )
            .to_result()?;

            array_uninit.assume_init()
        };

        // SAFETY: We've checked validity of all parameters. The transfer is device-to-device, which
        // should not interact with host memory (thus Rust memory safety) anyway.
        unsafe {
            cudart::cudaMemcpy2DToArray(
                array,                                       // dst
                0,                                           // width offset
                0,                                           // height offset
                cuda_slice.buffer,                           // src
                cuda_slice.pitch,                            // src pitch
                size.width as usize * BYTES_PER_RGBA8_PIXEL, // width in bytes
                size.height as usize,                        // height in rows
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,    // memcpy kind
            )
            .to_result()?;
        }

        // SAFETY: this is symmetrical to cudaGraphicsMapResources() call above.
        unsafe {
            cudart::cudaGraphicsUnmapResources(
                1,
                &mut registered_texture.graphics_resource,
                std::ptr::null_mut(),
            )
            .to_result()?;
        }

        Ok(())
    }
}

/// Extension trait for `cudaError_t` to convert it to Rust error.
trait CudaErrorTExt {
    fn to_result(self) -> Result<()>;
}

impl CudaErrorTExt for cudart::cudaError_t {
    fn to_result(self) -> Result<()> {
        match self {
            cudart::cudaError_t::cudaSuccess => Ok(()),
            other => Err(anyhow!("Got CUDA error {other:?}")),
        }
    }
}

extern "C" {
    pub fn cudaGraphicsGLRegisterImage(
        resource: *mut cudart::cudaGraphicsResource_t,
        texture_id: u32,
        target: u32,
        flags: u32,
    ) -> cudart::cudaError_t;
}
