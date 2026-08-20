use std::{
    collections::{BTreeSet, HashMap},
    env,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    fs::{self, File},
    path::{Path, PathBuf},
    ptr::NonNull,
};

use anyhow::{Context, Result, bail, ensure};
use catena_lang::{codegen::GpuDialect, runtime::MemRef};
use libloading::{Library, Symbol};
use memmap2::{Mmap, MmapOptions};

use crate::spec::{TensorSpec, WEIGHT_COUNT, WeightDtype, weight_specs};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: usize = 32;
const HIP_HOST_REGISTER_MAPPED: c_uint = 0x2;
// AMD-specific: immutable weights need only kernel-boundary coherence and can
// use the cache-friendly coarse-grained mapping.
const HIP_EXT_HOST_REGISTER_COARSE_GRAINED: c_uint = 0x8;

type HipHostRegister = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
type HipHostGetDevicePointer = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> c_int;
type HipHostUnregister = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32,
    Bf16,
}

impl GgufDtype {
    fn from_ggml_type(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            30 => Ok(Self::Bf16),
            other => bail!("unsupported GGUF tensor type {other}; expected F32 or BF16"),
        }
    }

    fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    dtype: GgufDtype,
    relative_offset: usize,
}

#[derive(Debug, Default)]
pub(crate) struct GgufTokenizerMetadata {
    pub(crate) model: Option<String>,
    pub(crate) pre: Option<String>,
    pub(crate) tokens: Vec<String>,
    pub(crate) merges: Vec<String>,
    pub(crate) token_types: Vec<i32>,
    pub(crate) special_token_ids: Vec<u32>,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_id: Option<u32>,
    pub(crate) add_bos_token: bool,
    pub(crate) add_eos_token: bool,
}

struct GgufFile {
    path: PathBuf,
    bytes: Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView {
    pub(crate) dtype: GgufDtype,
    file_index: usize,
    pub(crate) file_offset: usize,
    pub(crate) byte_len: usize,
}

/// Read-only GGUF mappings registered directly in the HIP address space.
pub struct Checkpoint {
    files: Vec<RegisteredGgufFile>,
    tensor_files: HashMap<String, usize>,
}

/// A registration must be dropped before its backing mapping.
struct RegisteredGgufFile {
    registration: HipRegistration,
    file: GgufFile,
}

impl Checkpoint {
    /// Open either one GGUF, any member of a standard GGUF shard set, or a
    /// directory containing GGUF shards.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let paths = resolve_gguf_paths(path.as_ref())?;
        let files = paths
            .iter()
            .map(GgufFile::open)
            .collect::<Result<Vec<_>>>()?;
        let tensor_files = validate_files(&files)?;
        let registered_files = files
            .into_iter()
            .map(|file| {
                let registration = HipRegistration::new(&file.bytes).with_context(|| {
                    format!("failed to register GGUF {} with HIP", file.path.display())
                })?;
                Ok(RegisteredGgufFile { registration, file })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            files: registered_files,
            tensor_files,
        })
    }

    pub fn tensor_view(&self, spec: &TensorSpec) -> Result<TensorView> {
        let file_index = *self
            .tensor_files
            .get(&spec.gguf_name)
            .with_context(|| format!("GGUF is missing {}", spec.gguf_name))?;
        self.files[file_index].file.tensor_view(spec, file_index)
    }

    /// Borrow one registered GGUF payload range as HIP-visible memory.
    pub fn mapped_ref<'a>(&'a self, view: &TensorView) -> Result<MemRef<'a>> {
        let end = view
            .file_offset
            .checked_add(view.byte_len)
            .context("GGUF device view range overflowed")?;
        let registered = self
            .files
            .get(view.file_index)
            .context("GGUF device view refers to an unknown shard")?;
        ensure!(
            end <= registered.file.bytes.len(),
            "GGUF device view is out of bounds"
        );
        // SAFETY: `registration` covers the entire live `Mmap`; the validated
        // range lies inside it, and the returned borrow is tied to `self`.
        let data = unsafe {
            registered
                .registration
                .device_base
                .as_ptr()
                .cast::<u8>()
                .add(view.file_offset)
                .cast::<c_void>()
        };
        // SAFETY: the HIP registration makes this range device-visible and is
        // held until `self` can no longer be borrowed.
        Ok(unsafe {
            MemRef::from_raw_parts(
                data,
                view.byte_len as u64,
                GpuDialect::Hip,
                &registered.registration,
            )
        })
    }

    pub fn registered_byte_len(&self) -> usize {
        self.files.iter().map(|file| file.file.bytes.len()).sum()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

fn resolve_gguf_paths(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_dir() {
        let mut paths = fs::read_dir(path)
            .with_context(|| format!("failed to read GGUF directory {}", path.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.retain(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        });
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "GGUF directory {} contains no .gguf files",
            path.display()
        );
        return Ok(paths);
    }

    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some(stem) = file_name.strip_suffix(".gguf") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((numbered_prefix, total_text)) = stem.rsplit_once("-of-") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((prefix, index_text)) = numbered_prefix.rsplit_once('-') else {
        return Ok(vec![path.to_path_buf()]);
    };
    if index_text.is_empty()
        || total_text.is_empty()
        || !index_text.bytes().all(|byte| byte.is_ascii_digit())
        || !total_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Ok(vec![path.to_path_buf()]);
    }
    let index = index_text
        .parse::<usize>()
        .context("invalid GGUF shard index")?;
    let total = total_text
        .parse::<usize>()
        .context("invalid GGUF shard count")?;
    ensure!(
        total > 0 && index > 0 && index <= total,
        "invalid GGUF shard {index} of {total} in {}",
        path.display()
    );
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let paths = (1..=total)
        .map(|shard| {
            parent.join(format!(
                "{prefix}-{shard:0width$}-of-{total_text}.gguf",
                width = index_text.len()
            ))
        })
        .collect();
    Ok(paths)
}

pub(crate) fn read_gguf_tokenizer_metadata(
    path: impl AsRef<Path>,
) -> Result<Option<GgufTokenizerMetadata>> {
    let paths = resolve_gguf_paths(path.as_ref())?;
    let path = paths
        .first()
        .context("GGUF path resolution returned no files")?;
    let file = File::open(path)
        .with_context(|| format!("failed to open GGUF tokenizer source {}", path.display()))?;
    // SAFETY: only metadata is read from this temporary read-only mapping.
    let bytes = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to map GGUF tokenizer source {}", path.display()))?;
    parse_gguf_tokenizer_metadata(&bytes)
        .with_context(|| format!("failed to parse GGUF tokenizer metadata {}", path.display()))
}

fn validate_files(files: &[GgufFile]) -> Result<HashMap<String, usize>> {
    let mut tensor_files = HashMap::with_capacity(WEIGHT_COUNT);
    for (file_index, file) in files.iter().enumerate() {
        for name in file.tensors.keys() {
            if let Some(previous_index) = tensor_files.insert(name.clone(), file_index) {
                bail!(
                    "duplicate GGUF tensor {name} in {} and {}",
                    files[previous_index].path.display(),
                    file.path.display()
                );
            }
        }
    }

    ensure!(
        tensor_files.len() == WEIGHT_COUNT,
        "GGUF shard set contains {} tensors, expected {WEIGHT_COUNT}",
        tensor_files.len()
    );
    let specs = weight_specs();
    let expected_names = specs
        .iter()
        .map(|spec| spec.gguf_name.as_str())
        .collect::<BTreeSet<_>>();
    let actual_names = tensor_files
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if expected_names != actual_names {
        let missing = expected_names
            .difference(&actual_names)
            .take(16)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = actual_names
            .difference(&expected_names)
            .take(16)
            .copied()
            .collect::<Vec<_>>();
        bail!(
            "GGUF tensor inventory mismatch; missing(first 16)={missing:?}, unexpected(first 16)={unexpected:?}"
        );
    }

    for (file_index, file) in files.iter().enumerate() {
        let mut ranges = Vec::new();
        for spec in specs
            .iter()
            .filter(|spec| tensor_files[&spec.gguf_name] == file_index)
        {
            let view = file.tensor_view(spec, file_index)?;
            ranges.push((
                view.file_offset,
                view.file_offset + view.byte_len,
                &spec.gguf_name,
            ));
        }
        ranges.sort_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            ensure!(
                pair[0].1 <= pair[1].0,
                "GGUF tensors {} and {} overlap in {}",
                pair[0].2,
                pair[1].2,
                file.path.display()
            );
        }
    }
    Ok(tensor_files)
}

impl GgufFile {
    fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("failed to open GGUF checkpoint {}", path.display()))?;
        // SAFETY: this is a read-only mapping and the model file must remain
        // immutable for the lifetime of the loaded model.
        let bytes = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to map GGUF checkpoint {}", path.display()))?;
        let (data_start, tensors) = parse_gguf(&bytes)
            .with_context(|| format!("failed to parse GGUF checkpoint {}", path.display()))?;
        Ok(Self {
            path,
            bytes,
            data_start,
            tensors,
        })
    }

    fn tensor_view(&self, spec: &TensorSpec, file_index: usize) -> Result<TensorView> {
        let info = self
            .tensors
            .get(&spec.gguf_name)
            .with_context(|| format!("GGUF is missing {}", spec.gguf_name))?;
        ensure!(
            info.shape == spec.shape,
            "GGUF tensor {} has shape {:?}, expected {:?} for {}",
            spec.gguf_name,
            info.shape,
            spec.shape,
            spec.name
        );
        let expected_dtype = match spec.dtype {
            WeightDtype::F32 => GgufDtype::F32,
            WeightDtype::Bf16 => GgufDtype::Bf16,
        };
        ensure!(
            info.dtype == expected_dtype,
            "GGUF tensor {} has dtype {:?}, expected {:?} for {}",
            spec.gguf_name,
            info.dtype,
            expected_dtype,
            spec.name
        );
        let element_count = spec
            .shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .with_context(|| format!("tensor {} element count overflowed", spec.gguf_name))?;
        let byte_len = element_count
            .checked_mul(info.dtype.element_size())
            .with_context(|| format!("tensor {} byte count overflowed", spec.gguf_name))?;
        let file_offset = self
            .data_start
            .checked_add(info.relative_offset)
            .with_context(|| format!("tensor {} file offset overflowed", spec.gguf_name))?;
        let end = file_offset
            .checked_add(byte_len)
            .with_context(|| format!("tensor {} range overflowed", spec.gguf_name))?;
        ensure!(
            end <= self.bytes.len(),
            "GGUF tensor {} range {file_offset}..{end} exceeds file length {}",
            spec.gguf_name,
            self.bytes.len()
        );
        Ok(TensorView {
            dtype: info.dtype,
            file_index,
            file_offset,
            byte_len,
        })
    }
}

struct HipRegistration {
    host_base: NonNull<c_void>,
    device_base: NonNull<c_void>,
    unregister: HipHostUnregister,
    error_string: HipGetErrorString,
    _library: Library,
}

impl HipRegistration {
    fn new(bytes: &Mmap) -> Result<Self> {
        ensure!(!bytes.is_empty(), "cannot register an empty GGUF mapping");
        let library = load_hip_library()?;
        let register: HipHostRegister = load_symbol(&library, b"hipHostRegister\0")?;
        let get_device_pointer: HipHostGetDevicePointer =
            load_symbol(&library, b"hipHostGetDevicePointer\0")?;
        let unregister: HipHostUnregister = load_symbol(&library, b"hipHostUnregister\0")?;
        let error_string: HipGetErrorString = load_symbol(&library, b"hipGetErrorString\0")?;
        let host_base = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .context("GGUF mmap returned a null pointer")?;
        let flags = HIP_HOST_REGISTER_MAPPED | HIP_EXT_HOST_REGISTER_COARSE_GRAINED;
        // SAFETY: the read-only mapping remains live in `Checkpoint` until it
        // is unregistered, and HIP permits registering existing host memory.
        let status = unsafe { register(host_base.as_ptr(), bytes.len(), flags) };
        if status != 0 {
            bail!(
                "hipHostRegister failed: {}",
                hip_error(error_string, status)
            );
        }

        let mut device_base = std::ptr::null_mut();
        // SAFETY: `host_base` was successfully registered immediately above.
        let status = unsafe { get_device_pointer(&mut device_base, host_base.as_ptr(), 0) };
        if status != 0 {
            // SAFETY: registration succeeded and must be undone on this error path.
            unsafe { unregister(host_base.as_ptr()) };
            bail!(
                "hipHostGetDevicePointer failed: {}",
                hip_error(error_string, status)
            );
        }
        let Some(device_base) = NonNull::new(device_base) else {
            // SAFETY: registration succeeded and must be undone on this error path.
            unsafe { unregister(host_base.as_ptr()) };
            bail!("hipHostGetDevicePointer returned a null pointer");
        };

        Ok(Self {
            host_base,
            device_base,
            unregister,
            error_string,
            _library: library,
        })
    }
}

impl Drop for HipRegistration {
    fn drop(&mut self) {
        // All current Runtime calls synchronize before returning, so no kernel
        // may still be reading the mapping when the model is dropped.
        let status = unsafe { (self.unregister)(self.host_base.as_ptr()) };
        if status != 0 {
            eprintln!(
                "warning: hipHostUnregister failed while dropping GGUF mapping: {}",
                hip_error(self.error_string, status)
            );
        }
    }
}

fn load_hip_library() -> Result<Library> {
    let mut paths = vec![PathBuf::from("libamdhip64.so")];
    for variable in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = env::var_os(variable) {
            let path = PathBuf::from(root).join("lib/libamdhip64.so");
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }

    let mut errors = Vec::new();
    for path in &paths {
        // SAFETY: this is the official HIP runtime library, and every resolved
        // symbol is invoked with its documented ABI below.
        match unsafe { Library::new(path) } {
            Ok(library) => return Ok(library),
            Err(error) => errors.push(format!("{}: {error}", path.display())),
        }
    }
    bail!("failed to load HIP runtime library ({})", errors.join(", "))
}

fn load_symbol<T: Copy>(library: &Library, symbol: &'static [u8]) -> Result<T> {
    // SAFETY: callers request the exact documented HIP function signature.
    let loaded: Symbol<'_, T> = unsafe { library.get(symbol) }.with_context(|| {
        format!(
            "failed to resolve HIP symbol {}",
            String::from_utf8_lossy(symbol).trim_end_matches('\0')
        )
    })?;
    Ok(*loaded)
}

fn hip_error(error_string: HipGetErrorString, status: c_int) -> String {
    // SAFETY: HIP returns a static NUL-terminated string for an error code.
    let pointer = unsafe { error_string(status) };
    if pointer.is_null() {
        format!("status {status}")
    } else {
        // SAFETY: checked non-null and guaranteed NUL-terminated by HIP.
        format!(
            "{} (status {status})",
            unsafe { CStr::from_ptr(pointer) }.to_string_lossy()
        )
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

fn parse_gguf_tokenizer_metadata(bytes: &[u8]) -> Result<Option<GgufTokenizerMetadata>> {
    ensure!(
        cfg!(target_endian = "little"),
        "GGUF tokenizer parsing requires a little-endian host"
    );
    let mut cursor = Cursor::new(bytes);
    ensure!(cursor.u32()? == GGUF_MAGIC, "file does not have GGUF magic");
    ensure!(
        cursor.u32()? == GGUF_VERSION,
        "only GGUF version {GGUF_VERSION} is supported"
    );
    let _tensor_count = cursor.usize_u64("tensor count")?;
    let metadata_count = cursor.usize_u64("metadata count")?;
    ensure!(
        metadata_count <= 1_000_000,
        "unreasonable GGUF metadata count {metadata_count}"
    );

    let mut metadata = GgufTokenizerMetadata::default();
    for _ in 0..metadata_count {
        let key = cursor.string("metadata key")?.to_owned();
        let value_type = ValueType::from_u32(cursor.u32()?)?;
        match key.as_str() {
            "tokenizer.ggml.model" => {
                metadata.model = Some(cursor.string_value(value_type, &key)?.to_owned());
            }
            "tokenizer.ggml.pre" => {
                metadata.pre = Some(cursor.string_value(value_type, &key)?.to_owned());
            }
            "tokenizer.ggml.tokens" => {
                metadata.tokens = cursor.string_array(value_type, &key)?;
            }
            "tokenizer.ggml.merges" => {
                metadata.merges = cursor.string_array(value_type, &key)?;
            }
            "tokenizer.ggml.token_type" => {
                metadata.token_types = cursor.i32_array(value_type, &key)?;
            }
            "tokenizer.ggml.bos_token_id" => {
                let id = cursor.u32_value(value_type, &key)?;
                metadata.bos_token_id = Some(id);
                metadata.special_token_ids.push(id);
            }
            "tokenizer.ggml.eos_token_id" => {
                let id = cursor.u32_value(value_type, &key)?;
                metadata.eos_token_id = Some(id);
                metadata.special_token_ids.push(id);
            }
            "tokenizer.ggml.unknown_token_id"
            | "tokenizer.ggml.separator_token_id"
            | "tokenizer.ggml.padding_token_id"
            | "tokenizer.ggml.mask_token_id" => {
                metadata
                    .special_token_ids
                    .push(cursor.u32_value(value_type, &key)?);
            }
            "tokenizer.ggml.add_bos_token" => {
                metadata.add_bos_token = cursor.bool_value(value_type, &key)?;
            }
            "tokenizer.ggml.add_eos_token" => {
                metadata.add_eos_token = cursor.bool_value(value_type, &key)?;
            }
            _ => cursor.skip_value(value_type, 0)?,
        }
    }
    Ok((!metadata.tokens.is_empty()).then_some(metadata))
}

impl ValueType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            other => bail!("unrecognized GGUF metadata value type {other}"),
        }
    }
}

fn parse_gguf(bytes: &[u8]) -> Result<(usize, HashMap<String, TensorInfo>)> {
    ensure!(
        cfg!(target_endian = "little"),
        "GGUF zero-copy requires a little-endian host"
    );
    let mut cursor = Cursor::new(bytes);
    ensure!(cursor.u32()? == GGUF_MAGIC, "file does not have GGUF magic");
    ensure!(
        cursor.u32()? == GGUF_VERSION,
        "only GGUF version {GGUF_VERSION} is supported"
    );
    let tensor_count = cursor.usize_u64("tensor count")?;
    let metadata_count = cursor.usize_u64("metadata count")?;
    ensure!(
        tensor_count <= 1_000_000,
        "unreasonable GGUF tensor count {tensor_count}"
    );
    ensure!(
        metadata_count <= 1_000_000,
        "unreasonable GGUF metadata count {metadata_count}"
    );

    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let key = cursor.string("metadata key")?;
        let value_type = ValueType::from_u32(cursor.u32()?)?;
        if key == "general.alignment" {
            alignment = cursor.unsigned_value(value_type, "general.alignment")?;
        } else {
            cursor.skip_value(value_type, 0)?;
        }
    }
    ensure!(
        alignment > 0 && alignment.is_power_of_two(),
        "GGUF alignment {alignment} is not a positive power of two"
    );

    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.string("tensor name")?.to_owned();
        let dimension_count = cursor.u32()? as usize;
        ensure!(
            dimension_count <= 8,
            "tensor {name} has {dimension_count} dimensions"
        );
        let mut shape = Vec::with_capacity(dimension_count);
        for _ in 0..dimension_count {
            shape.push(cursor.usize_u64("tensor dimension")?);
        }
        shape.reverse();
        let dtype = GgufDtype::from_ggml_type(cursor.u32()?)?;
        let relative_offset = cursor.usize_u64("tensor offset")?;
        ensure!(
            relative_offset.is_multiple_of(alignment),
            "tensor {name} offset {relative_offset} is not aligned to {alignment}"
        );
        ensure!(
            tensors
                .insert(
                    name.clone(),
                    TensorInfo {
                        shape,
                        dtype,
                        relative_offset,
                    },
                )
                .is_none(),
            "duplicate GGUF tensor name {name}"
        );
    }
    let data_start = cursor
        .position()
        .checked_add(alignment - 1)
        .map(|position| position & !(alignment - 1))
        .context("GGUF data alignment overflowed")?;
    ensure!(
        data_start <= bytes.len(),
        "GGUF tensor data begins beyond end of file"
    );
    Ok((data_start, tensors))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .context("GGUF cursor overflowed")?;
        let output = self
            .bytes
            .get(self.position..end)
            .context("unexpected end of GGUF metadata")?;
        self.position = end;
        Ok(output)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn usize_u64(&mut self, label: &str) -> Result<usize> {
        usize::try_from(self.u64()?).with_context(|| format!("GGUF {label} exceeds usize"))
    }

    fn string(&mut self, label: &str) -> Result<&'a str> {
        let len = self.usize_u64(&format!("{label} length"))?;
        std::str::from_utf8(self.take(len)?).with_context(|| format!("GGUF {label} is not UTF-8"))
    }

    fn string_value(&mut self, value_type: ValueType, label: &str) -> Result<&'a str> {
        ensure!(
            matches!(value_type, ValueType::String),
            "GGUF {label} is not a string"
        );
        self.string(label)
    }

    fn u32_value(&mut self, value_type: ValueType, label: &str) -> Result<u32> {
        let value = match value_type {
            ValueType::U8 => u64::from(self.u8()?),
            ValueType::U16 => u64::from(self.u16()?),
            ValueType::U32 => u64::from(self.u32()?),
            ValueType::U64 => self.u64()?,
            _ => bail!("GGUF {label} is not an unsigned integer"),
        };
        u32::try_from(value).with_context(|| format!("GGUF {label} exceeds u32"))
    }

    fn bool_value(&mut self, value_type: ValueType, label: &str) -> Result<bool> {
        ensure!(
            matches!(value_type, ValueType::Bool),
            "GGUF {label} is not a boolean"
        );
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => bail!("GGUF {label} has invalid boolean value {value}"),
        }
    }

    fn array_header(&mut self, value_type: ValueType, label: &str) -> Result<(ValueType, usize)> {
        ensure!(
            matches!(value_type, ValueType::Array),
            "GGUF {label} is not an array"
        );
        let element_type = ValueType::from_u32(self.u32()?)?;
        let len = self.usize_u64(&format!("{label} array length"))?;
        ensure!(
            len <= 10_000_000,
            "unreasonable GGUF {label} array length {len}"
        );
        Ok((element_type, len))
    }

    fn string_array(&mut self, value_type: ValueType, label: &str) -> Result<Vec<String>> {
        let (element_type, len) = self.array_header(value_type, label)?;
        ensure!(
            matches!(element_type, ValueType::String),
            "GGUF {label} is not a string array"
        );
        (0..len)
            .map(|_| self.string(label).map(str::to_owned))
            .collect()
    }

    fn i32_array(&mut self, value_type: ValueType, label: &str) -> Result<Vec<i32>> {
        let (element_type, len) = self.array_header(value_type, label)?;
        ensure!(
            matches!(element_type, ValueType::I32),
            "GGUF {label} is not an i32 array"
        );
        (0..len)
            .map(|_| Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap())))
            .collect()
    }

    fn unsigned_value(&mut self, value_type: ValueType, label: &str) -> Result<usize> {
        let value = match value_type {
            ValueType::U8 => u64::from(self.u8()?),
            ValueType::U16 => u64::from(self.u16()?),
            ValueType::U32 => u64::from(self.u32()?),
            ValueType::U64 => self.u64()?,
            _ => bail!("GGUF {label} is not an unsigned integer"),
        };
        usize::try_from(value).with_context(|| format!("GGUF {label} exceeds usize"))
    }

    fn skip_value(&mut self, value_type: ValueType, depth: usize) -> Result<()> {
        ensure!(depth <= 8, "GGUF metadata arrays are nested too deeply");
        match value_type {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => {
                self.take(1)?;
            }
            ValueType::U16 | ValueType::I16 => {
                self.take(2)?;
            }
            ValueType::U32 | ValueType::I32 | ValueType::F32 => {
                self.take(4)?;
            }
            ValueType::U64 | ValueType::I64 | ValueType::F64 => {
                self.take(8)?;
            }
            ValueType::String => {
                let len = self.usize_u64("metadata string length")?;
                self.take(len)?;
            }
            ValueType::Array => {
                let element_type = ValueType::from_u32(self.u32()?)?;
                let len = self.usize_u64("metadata array length")?;
                ensure!(
                    len <= 10_000_000,
                    "unreasonable GGUF metadata array length {len}"
                );
                for _ in 0..len {
                    self.skip_value(element_type, depth + 1)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    const GGUF_PATH: &str = "models/Qwen3-30B-A3B-Base/model-bf16.gguf";

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend(value.as_bytes());
    }

    fn push_metadata_key(bytes: &mut Vec<u8>, key: &str, value_type: ValueType) {
        push_string(bytes, key);
        push_u32(bytes, value_type as u32);
    }

    fn push_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_metadata_key(bytes, key, ValueType::String);
        push_string(bytes, value);
    }

    fn push_string_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        push_metadata_key(bytes, key, ValueType::Array);
        push_u32(bytes, ValueType::String as u32);
        push_u64(bytes, values.len() as u64);
        for value in values {
            push_string(bytes, value);
        }
    }

    #[test]
    fn resolves_every_file_from_any_standard_shard() -> Result<()> {
        let paths = resolve_gguf_paths(Path::new("models/Qwen3-30B-A3B-BF16-00002-of-00003.gguf"))?;
        assert_eq!(
            paths,
            [
                "models/Qwen3-30B-A3B-BF16-00001-of-00003.gguf",
                "models/Qwen3-30B-A3B-BF16-00002-of-00003.gguf",
                "models/Qwen3-30B-A3B-BF16-00003-of-00003.gguf",
            ]
            .map(PathBuf::from)
        );
        Ok(())
    }

    #[test]
    fn leaves_an_unsharded_name_alone() -> Result<()> {
        let path = Path::new("models/model-bf16.gguf");
        assert_eq!(resolve_gguf_paths(path)?, [path]);
        Ok(())
    }

    #[test]
    fn resolves_a_directory_to_sorted_gguf_files_only() -> Result<()> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "catena-qwen3-gguf-paths-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        for name in ["part-00002.gguf", "tokenizer.json", "part-00001.gguf"] {
            File::create(directory.join(name))?;
        }
        let paths = resolve_gguf_paths(&directory)?;
        fs::remove_dir_all(&directory)?;
        assert_eq!(
            paths,
            ["part-00001.gguf", "part-00002.gguf"].map(|name| directory.join(name))
        );
        Ok(())
    }

    #[test]
    fn parses_embedded_tokenizer_metadata_without_tensor_data() -> Result<()> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, GGUF_MAGIC);
        push_u32(&mut bytes, GGUF_VERSION);
        push_u64(&mut bytes, 42);
        push_u64(&mut bytes, 8);
        push_string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2");
        push_string_metadata(&mut bytes, "tokenizer.ggml.pre", "qwen2");
        push_string_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &["h", "i", "hi"]);
        push_string_array_metadata(&mut bytes, "tokenizer.ggml.merges", &["h i"]);
        push_metadata_key(&mut bytes, "tokenizer.ggml.token_type", ValueType::Array);
        push_u32(&mut bytes, ValueType::I32 as u32);
        push_u64(&mut bytes, 3);
        for token_type in [1_i32, 1, 3] {
            bytes.extend(token_type.to_le_bytes());
        }
        push_metadata_key(&mut bytes, "tokenizer.ggml.eos_token_id", ValueType::U32);
        push_u32(&mut bytes, 2);
        push_metadata_key(&mut bytes, "tokenizer.ggml.add_eos_token", ValueType::Bool);
        bytes.push(1);
        push_string_metadata(&mut bytes, "unrelated.metadata", "ignored");

        let metadata =
            parse_gguf_tokenizer_metadata(&bytes)?.context("tokenizer metadata was not found")?;
        assert_eq!(metadata.model.as_deref(), Some("gpt2"));
        assert_eq!(metadata.pre.as_deref(), Some("qwen2"));
        assert_eq!(metadata.tokens, ["h", "i", "hi"]);
        assert_eq!(metadata.merges, ["h i"]);
        assert_eq!(metadata.token_types, [1, 1, 3]);
        assert_eq!(metadata.eos_token_id, Some(2));
        assert!(metadata.add_eos_token);
        Ok(())
    }

    #[test]
    fn validates_downloaded_gguf_when_present() -> Result<()> {
        let path = Path::new(GGUF_PATH);
        if path.exists() {
            let file = GgufFile::open(path)?;
            assert_eq!(file.tensors.len(), WEIGHT_COUNT);
            assert_eq!(file.data_start, 5_969_472);
            let specs = weight_specs();
            for spec in &specs {
                file.tensor_view(spec, 0)?;
            }
            let embedding = file.tensor_view(&specs[0], 0)?;
            assert_eq!(embedding.dtype, GgufDtype::Bf16);
            assert_eq!(embedding.byte_len, 151_936 * 2_048 * 2);
            let norm = file.tensor_view(&specs[1], 0)?;
            assert_eq!(norm.dtype, GgufDtype::F32);
            assert_eq!(norm.byte_len, 2_048 * 4);
        }
        Ok(())
    }
}
