use bytes::Bytes;
use mcp_protocol::{
    ZeroCopyBytes, ZeroCopyMessage, MmapHeader, BufferPool, 
    ProtocolState, Uninitialized,
};
use proptest::prelude::*;

#[test]
fn test_zero_copy_bytes() {
    let data = vec![1, 2, 3, 4, 5];
    let zc_bytes = ZeroCopyBytes::from_vec(data.clone());
    
    assert_eq!(zc_bytes.len(), 5);
    assert_eq!(zc_bytes.as_bytes(), &[1, 2, 3, 4, 5]);
    assert!(!zc_bytes.is_empty());
    
    let bytes = zc_bytes.into_bytes();
    assert_eq!(&bytes[..], &[1, 2, 3, 4, 5]);
}

#[test]
fn test_zero_copy_message_serialization() {
    let msg = ZeroCopyMessage {
        id: [1; 16],
        tool: 1,
        payload: vec![10, 20, 30],
    };
    
    let serialized = msg.serialize().unwrap();
    assert!(!serialized.is_empty());
    
    let deserialized = ZeroCopyMessage::deserialize(&serialized).unwrap();
    assert_eq!(deserialized.id, [1; 16]);
    assert_eq!(deserialized.tool, 1);
    assert_eq!(&deserialized.payload[..], &[10, 20, 30]);
}

#[test]
fn test_zero_copy_message_validation() {
    let msg = ZeroCopyMessage {
        id: [2; 16],
        tool: 2,
        payload: vec![40, 50],
    };
    
    let serialized = msg.serialize().unwrap();
    let validated = ZeroCopyMessage::deserialize_validated(&serialized).unwrap();
    
    assert_eq!(validated.id, [2; 16]);
    assert_eq!(validated.tool, 2);
}

#[test]
fn test_mmap_header() {
    let header = MmapHeader::new();
    
    assert_eq!(header.magic, MmapHeader::MAGIC);
    assert_eq!(header.version, MmapHeader::VERSION);
    assert_eq!(header.message_count, 0);
    assert!(header.validate().is_ok());
    
    // Test invalid header
    let mut invalid_header = header;
    invalid_header.magic = [0; 4];
    assert!(invalid_header.validate().is_err());
}

#[test]
fn test_buffer_pool() {
    let pool = BufferPool::new(1024, 2);
    
    // Acquire buffers
    let mut buf1 = pool.acquire();
    let mut buf2 = pool.acquire();
    let mut buf3 = pool.acquire(); // Should create new
    
    assert_eq!(buf1.capacity(), 1024);
    assert_eq!(buf2.capacity(), 1024);
    assert_eq!(buf3.capacity(), 1024);
    
    // Use buffers
    buf1.extend_from_slice(b"hello");
    buf2.extend_from_slice(b"world");
    buf3.extend_from_slice(b"test");
    
    // Release buffers
    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);
    
    // Reacquire should get cleared buffers
    let buf4 = pool.acquire();
    assert_eq!(buf4.len(), 0);
    assert!(buf4.capacity() >= 1024);
}

#[test]
fn test_type_state_pattern() {
    // Start with uninitialized state
    let state: ProtocolState<Uninitialized> = ProtocolState::new();
    
    // Can only initialize from uninitialized
    let initialized = state.initialize();
    
    // Can only connect from initialized
    let connected = initialized.connect();
    
    // Can disconnect and reconnect
    let disconnected = connected.disconnect();
    let _reconnected = disconnected.connect();
    
    // This ensures compile-time state machine correctness
}

// Property-based tests
proptest! {
    #[test]
    fn test_zero_copy_message_roundtrip(
        id in prop::array::uniform16(any::<u8>()),
        tool in any::<u8>(),
        payload in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        let msg = ZeroCopyMessage { id, tool, payload: payload.clone() };
        
        let serialized = msg.serialize().unwrap();
        let deserialized = ZeroCopyMessage::deserialize(&serialized).unwrap();
        
        prop_assert_eq!(deserialized.id, id);
        prop_assert_eq!(deserialized.tool, tool);
        prop_assert_eq!(&deserialized.payload[..], &payload[..]);
    }
    
    #[test]
    fn test_buffer_pool_stress(
        operations in prop::collection::vec(0..3u8, 10..100)
    ) {
        let pool = BufferPool::new(512, 4);
        let mut buffers = Vec::new();
        
        for op in operations {
            match op {
                0 => {
                    // Acquire
                    if buffers.len() < 10 {
                        buffers.push(pool.acquire());
                    }
                }
                1 => {
                    // Release
                    if let Some(buf) = buffers.pop() {
                        pool.release(buf);
                    }
                }
                2 => {
                    // Use buffer
                    if let Some(buf) = buffers.last_mut() {
                        buf.extend_from_slice(b"data");
                    }
                }
                _ => unreachable!(),
            }
        }
        
        // All operations should succeed without panic
        prop_assert!(true);
    }
}