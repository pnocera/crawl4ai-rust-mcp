use storage::{MmapStorage, AppendLog, RingBuffer};
use tempfile::tempdir;
use tokio::test;

#[test]
fn test_mmap_storage_create_and_write() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.mmap");
    
    let storage = MmapStorage::create(&path, 4096).unwrap();
    
    // Write at specific offset
    let data = b"Hello, mmap!";
    storage.write_bytes(100, data).unwrap();
    
    // Read back
    let read_data = storage.read_bytes(100, data.len()).unwrap();
    assert_eq!(&read_data[..], data);
    
    // Test append
    let offset = storage.append(b"Appended").unwrap();
    assert!(offset > 0);
    
    let appended = storage.read_bytes(offset, 8).unwrap();
    assert_eq!(&appended[..], b"Appended");
}

#[test]
fn test_mmap_storage_grow() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("grow.mmap");
    
    // Start with small file
    let storage = MmapStorage::create(&path, 1024).unwrap();
    
    // Write data that requires growth
    let large_data = vec![0u8; 2048];
    storage.write_bytes(0, &large_data).unwrap();
    
    // Should be able to read it back
    let read_data = storage.read_bytes(0, large_data.len()).unwrap();
    assert_eq!(read_data.len(), large_data.len());
}

#[test]
fn test_append_log() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("append.log");
    
    let log = AppendLog::create(&path).unwrap();
    
    // Append multiple entries
    let id1 = log.append(b"First entry").unwrap();
    let id2 = log.append(b"Second entry").unwrap();
    let id3 = log.append(b"Third entry").unwrap();
    
    // Read individual entries
    assert_eq!(&log.read(id1).unwrap()[..], b"First entry");
    assert_eq!(&log.read(id2).unwrap()[..], b"Second entry");
    assert_eq!(&log.read(id3).unwrap()[..], b"Third entry");
    
    // Iterate all entries
    let all_entries = log.iter().unwrap();
    assert_eq!(all_entries.len(), 3);
    assert_eq!(&all_entries[0].1[..], b"First entry");
    assert_eq!(&all_entries[1].1[..], b"Second entry");
    assert_eq!(&all_entries[2].1[..], b"Third entry");
}

#[test]
fn test_ring_buffer_basic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ring.buf");
    
    let ring = RingBuffer::create(&path, 100).unwrap();
    
    // Write data
    ring.write(b"Hello").unwrap();
    ring.write(b" World").unwrap();
    
    // Read data
    let mut buf = vec![0; 11];
    let n = ring.read(&mut buf).unwrap();
    assert_eq!(n, 11);
    assert_eq!(&buf[..n], b"Hello World");
    
    // Buffer should be empty now
    let n = ring.read(&mut buf).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn test_ring_buffer_wrap_around() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ring_wrap.buf");
    
    let ring = RingBuffer::create(&path, 10).unwrap();
    
    // Fill buffer
    ring.write(b"12345").unwrap();
    ring.write(b"67890").unwrap();
    
    // Read partial
    let mut buf = vec![0; 5];
    let n = ring.read(&mut buf).unwrap();
    assert_eq!(n, 5);
    assert_eq!(&buf[..n], b"12345");
    
    // Write more (should wrap)
    ring.write(b"ABCDE").unwrap();
    
    // Read remaining
    let mut buf = vec![0; 10];
    let n = ring.read(&mut buf).unwrap();
    assert_eq!(n, 10);
    assert_eq!(&buf[..n], b"67890ABCDE");
}

#[test]
fn test_ring_buffer_overflow_protection() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ring_overflow.buf");
    
    let ring = RingBuffer::create(&path, 10).unwrap();
    
    // Try to write more than capacity
    let result = ring.write(b"12345678901");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_append_log() {
    use std::sync::Arc;
    
    let dir = tempdir().unwrap();
    let path = dir.path().join("concurrent.log");
    
    let log = Arc::new(AppendLog::create(&path).unwrap());
    
    // Spawn multiple tasks to append concurrently
    let mut handles = vec![];
    
    for i in 0..10 {
        let log_clone = log.clone();
        let handle = tokio::spawn(async move {
            let data = format!("Entry from task {}", i);
            log_clone.append(data.as_bytes()).unwrap()
        });
        handles.push(handle);
    }
    
    // Wait for all tasks
    let mut ids = vec![];
    for handle in handles {
        ids.push(handle.await.unwrap());
    }
    
    // Verify all entries exist
    for (i, id) in ids.iter().enumerate() {
        let data = log.read(*id).unwrap();
        let expected = format!("Entry from task {}", i);
        assert_eq!(String::from_utf8_lossy(&data), expected);
    }
}