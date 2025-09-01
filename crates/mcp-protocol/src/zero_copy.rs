use bytes::{Bytes, BytesMut};
use rkyv::{
    Archive, Deserialize, Serialize,
    to_bytes, from_bytes, access,
};
use std::sync::Arc;
use zerocopy::{IntoBytes, FromBytes, FromZeros};

use crate::{ProtocolError, Result};

// Zero-copy wrapper for borrowed data
#[derive(Debug, Clone)]
pub struct ZeroCopyBytes {
    data: Bytes,
}

impl ZeroCopyBytes {
    pub fn new(data: Bytes) -> Self {
        Self { data }
    }

    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            data: Bytes::from(vec),
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn into_bytes(self) -> Bytes {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// Archive-friendly string type for zero-copy deserialization
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ArchivedString {
    inner: String,
}

impl From<String> for ArchivedString {
    fn from(s: String) -> Self {
        Self { inner: s }
    }
}

impl From<&str> for ArchivedString {
    fn from(s: &str) -> Self {
        Self {
            inner: s.to_string(),
        }
    }
}

// Zero-copy message container using rkyv
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ZeroCopyMessage {
    pub id: [u8; 16], // UUID as bytes
    pub tool: u8,     // Tool enum as byte
    pub payload: Vec<u8>,
}

impl ZeroCopyMessage {
    pub fn serialize(&self) -> Result<Bytes> {
        let bytes = to_bytes(self)
            .map_err(|e: rkyv::rancor::Error| ProtocolError::RkyvSerialization(e.to_string()))?;
        
        Ok(Bytes::from(bytes.into_vec()))
    }

    pub fn deserialize(bytes: &[u8]) -> Result<ZeroCopyMessage> {
        from_bytes(bytes)
            .map_err(|e: rkyv::rancor::Error| ProtocolError::ArchiveValidation(e.to_string()))
    }

    pub fn deserialize_validated(bytes: &[u8]) -> Result<ZeroCopyMessage> {
        // First validate the archived data
        let _archived = access::<ArchivedZeroCopyMessage, rkyv::rancor::Error>(bytes)
            .map_err(|e| ProtocolError::ArchiveValidation(e.to_string()))?;
        
        // Then deserialize 
        from_bytes(bytes)
            .map_err(|e: rkyv::rancor::Error| ProtocolError::ArchiveValidation(e.to_string()))
    }
}

// Memory-mapped message for zero-copy file operations
#[repr(C, packed)]
#[derive(IntoBytes, FromZeros, Debug, Clone, Copy)]
pub struct MmapHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub message_count: u32,
    pub total_size: u64,
}

impl MmapHeader {
    pub const MAGIC: [u8; 4] = *b"MCPR";
    pub const VERSION: u32 = 1;

    pub fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            message_count: 0,
            total_size: std::mem::size_of::<Self>() as u64,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(ProtocolError::InvalidFormat("Invalid magic bytes".to_string()));
        }
        if self.version != Self::VERSION {
            return Err(ProtocolError::InvalidFormat("Unsupported version".to_string()));
        }
        Ok(())
    }
}

// Buffer pool for zero-allocation message handling
pub struct BufferPool {
    pool: Arc<parking_lot::Mutex<Vec<BytesMut>>>,
    buffer_size: usize,
}

impl BufferPool {
    pub fn new(buffer_size: usize, initial_capacity: usize) -> Self {
        let mut pool = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            pool.push(BytesMut::with_capacity(buffer_size));
        }
        
        Self {
            pool: Arc::new(parking_lot::Mutex::new(pool)),
            buffer_size,
        }
    }

    pub fn acquire(&self) -> BytesMut {
        self.pool
            .lock()
            .pop()
            .unwrap_or_else(|| BytesMut::with_capacity(self.buffer_size))
    }

    pub fn release(&self, mut buffer: BytesMut) {
        buffer.clear();
        if buffer.capacity() <= self.buffer_size * 2 {
            self.pool.lock().push(buffer);
        }
    }
}

// Type-state pattern for compile-time protocol state tracking
pub struct Uninitialized;
pub struct Initialized;
pub struct Connected;

pub struct ProtocolState<S> {
    _state: std::marker::PhantomData<S>,
}

impl ProtocolState<Uninitialized> {
    pub fn new() -> Self {
        Self {
            _state: std::marker::PhantomData,
        }
    }

    pub fn initialize(self) -> ProtocolState<Initialized> {
        ProtocolState {
            _state: std::marker::PhantomData,
        }
    }
}

impl ProtocolState<Initialized> {
    pub fn connect(self) -> ProtocolState<Connected> {
        ProtocolState {
            _state: std::marker::PhantomData,
        }
    }
}

impl ProtocolState<Connected> {
    pub fn disconnect(self) -> ProtocolState<Initialized> {
        ProtocolState {
            _state: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_message_roundtrip() {
        let msg = ZeroCopyMessage {
            id: [1; 16],
            tool: 1,
            payload: vec![1, 2, 3, 4],
        };

        let serialized = msg.serialize().unwrap();
        let deserialized = ZeroCopyMessage::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.id, [1; 16]);
        assert_eq!(deserialized.tool, 1);
        assert_eq!(&deserialized.payload[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_mmap_header() {
        let header = MmapHeader::new();
        assert_eq!(header.magic, MmapHeader::MAGIC);
        assert_eq!(header.version, MmapHeader::VERSION);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(1024, 2);
        let buf1 = pool.acquire();
        let buf2 = pool.acquire();
        let buf3 = pool.acquire(); // Should create new buffer
        
        assert_eq!(buf1.capacity(), 1024);
        assert_eq!(buf2.capacity(), 1024);
        assert_eq!(buf3.capacity(), 1024);
        
        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3);
    }
}