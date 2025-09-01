use bytes::Bytes;
use memmap2::MmapMut;
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use zerocopy::{IntoBytes, FromBytes, Immutable};

use crate::{Result, StorageError};

// Memory-mapped file wrapper for zero-copy access
pub struct MmapStorage {
    path: PathBuf,
    file: File,
    mmap: Arc<RwLock<MmapMut>>,
    current_size: Arc<RwLock<usize>>,
}

impl MmapStorage {
    pub fn create(path: impl AsRef<Path>, initial_size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Create or open file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
        
        // Set initial file size
        file.set_len(initial_size as u64)?;
        
        // Create memory map
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(Self {
            path,
            file,
            mmap: Arc::new(RwLock::new(mmap)),
            current_size: Arc::new(RwLock::new(0)),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Read current size from header if exists
        let current_size = if file_size >= std::mem::size_of::<MmapFileHeader>() {
            let header = MmapFileHeader::read_from_bytes(&mmap[..std::mem::size_of::<MmapFileHeader>()]);
            header.map(|h| h.used_size as usize).unwrap_or(0)
        } else {
            0
        };
        
        Ok(Self {
            path,
            file,
            mmap: Arc::new(RwLock::new(mmap)),
            current_size: Arc::new(RwLock::new(current_size)),
        })
    }

    pub fn write_bytes(&self, offset: usize, data: &[u8]) -> Result<()> {
        let mut mmap = self.mmap.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();
        
        // Check if we need to grow the file
        let required_size = offset + data.len();
        if required_size > mmap.len() {
            self.grow_file(required_size)?;
            // Re-map after growing
            drop(mmap);
            let new_mmap = unsafe { MmapMut::map_mut(&self.file)? };
            mmap = self.mmap.write().unwrap();
            *mmap = new_mmap;
        }
        
        // Write data
        mmap[offset..offset + data.len()].copy_from_slice(data);
        
        // Update current size
        *current_size = (*current_size).max(offset + data.len());
        
        // Update header
        self.update_header(*current_size)?;
        
        Ok(())
    }

    pub fn read_bytes(&self, offset: usize, len: usize) -> Result<Bytes> {
        let mmap = self.mmap.read().unwrap();
        
        if offset + len > mmap.len() {
            return Err(StorageError::MemoryMap(
                "Read beyond end of mapped region".to_string()
            ));
        }
        
        // Zero-copy slice
        Ok(Bytes::copy_from_slice(&mmap[offset..offset + len]))
    }

    pub fn append(&self, data: &[u8]) -> Result<usize> {
        let current_size = *self.current_size.read().unwrap();
        let offset = current_size;
        self.write_bytes(offset, data)?;
        Ok(offset)
    }

    pub fn flush(&self) -> Result<()> {
        let mmap = self.mmap.read().unwrap();
        mmap.flush()?;
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        let mmap = self.mmap.read().unwrap();
        mmap.flush_async()?;
        Ok(())
    }

    fn grow_file(&self, new_size: usize) -> Result<()> {
        // Grow by at least 50% or to required size
        let current_len = self.file.metadata()?.len() as usize;
        let new_len = (current_len * 3 / 2).max(new_size);
        
        self.file.set_len(new_len as u64)?;
        Ok(())
    }

    fn update_header(&self, used_size: usize) -> Result<()> {
        let header = MmapFileHeader {
            magic: MmapFileHeader::MAGIC,
            version: MmapFileHeader::VERSION,
            used_size: used_size as u64,
            reserved: [0; 40],
        };
        
        let mut mmap = self.mmap.write().unwrap();
        mmap[..std::mem::size_of::<MmapFileHeader>()]
            .copy_from_slice(header.as_bytes());
        
        Ok(())
    }
}

// Zero-copy append-only log using memory-mapped files
pub struct AppendLog {
    storage: MmapStorage,
    index: Arc<RwLock<Vec<LogEntry>>>,
}

#[repr(C, packed)]
#[derive(IntoBytes, FromBytes, Immutable, Debug, Clone, Copy)]
struct MmapFileHeader {
    magic: [u8; 8],
    version: u32,
    used_size: u64,
    reserved: [u8; 40],
}

impl MmapFileHeader {
    const MAGIC: [u8; 8] = *b"MCPRMMAP";
    const VERSION: u32 = 1;
}

#[derive(Debug, Clone)]
struct LogEntry {
    offset: usize,
    len: usize,
    timestamp: u64,
}

impl AppendLog {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let storage = MmapStorage::create(
            path,
            1024 * 1024 * 10, // 10MB initial size
        )?;
        
        // Write header
        let header = MmapFileHeader {
            magic: MmapFileHeader::MAGIC,
            version: MmapFileHeader::VERSION,
            used_size: std::mem::size_of::<MmapFileHeader>() as u64,
            reserved: [0; 40],
        };
        
        storage.write_bytes(0, header.as_bytes())?;
        
        Ok(Self {
            storage,
            index: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = MmapStorage::open(path)?;
        
        // Rebuild index from log by scanning through all entries
        let index = Self::rebuild_index_from_storage(&storage)?;
        
        Ok(Self { storage, index })
    }
    
    fn rebuild_index_from_storage(storage: &MmapStorage) -> Result<Arc<RwLock<Vec<LogEntry>>>> {
        let mut index = Vec::new();
        let header_size = std::mem::size_of::<MmapFileHeader>();
        let entry_header_size = std::mem::size_of::<EntryHeader>();
        let current_size = *storage.current_size.read().unwrap();
        
        let mut offset = header_size;
        
        // Scan through the file to rebuild the index
        while offset + entry_header_size < current_size {
            // Read entry header
            let header_bytes = storage.read_bytes(offset, entry_header_size)?;
            if header_bytes.len() < entry_header_size {
                break; // End of valid data
            }
            
            let entry_header = EntryHeader::read_from_bytes(&header_bytes[..entry_header_size])
                .map_err(|_| StorageError::CorruptedData("Invalid entry header".to_string()))?;
            
            // Validate entry integrity
            let data_offset = offset + entry_header_size;
            let data_bytes = storage.read_bytes(data_offset, entry_header.len as usize)?;
            
            if data_bytes.len() != entry_header.len as usize {
                break; // Incomplete entry, probably end of file
            }
            
            // Verify checksum
            let calculated_checksum = Self::calculate_checksum(&data_bytes);
            let expected_checksum = entry_header.checksum; // Copy to avoid unaligned reference
            if calculated_checksum != expected_checksum {
                tracing::warn!(
                    "Checksum mismatch at offset {}: expected {}, got {}",
                    offset,
                    expected_checksum,
                    calculated_checksum
                );
                // Continue anyway but log the issue
            }
            
            // Add to index
            index.push(LogEntry {
                offset: data_offset, // Point to actual data, not header
                len: entry_header.len as usize,
                timestamp: entry_header.timestamp,
            });
            
            // Move to next entry
            offset = data_offset + entry_header.len as usize;
        }
        
        tracing::info!("Rebuilt index with {} entries from log file", index.len());
        Ok(Arc::new(RwLock::new(index)))
    }

    pub fn append(&self, data: &[u8]) -> Result<usize> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Calculate checksum for data integrity
        let checksum = Self::calculate_checksum(data);
        
        // Create entry header
        let entry_header = EntryHeader {
            len: data.len() as u32,
            timestamp,
            checksum,
        };
        
        // Calculate offset (after current header)
        let header_size = std::mem::size_of::<MmapFileHeader>();
        let current_size = *self.storage.current_size.read().unwrap();
        let offset = current_size.max(header_size);
        
        // Write entry header and data
        self.storage.write_bytes(offset, entry_header.as_bytes())?;
        self.storage.write_bytes(
            offset + std::mem::size_of::<EntryHeader>(),
            data
        )?;
        
        // Update index
        let entry_id = {
            let mut index = self.index.write().unwrap();
            let id = index.len();
            index.push(LogEntry {
                offset: offset + std::mem::size_of::<EntryHeader>(),
                len: data.len(),
                timestamp,
            });
            id
        };
        
        Ok(entry_id)
    }

    pub fn read(&self, entry_id: usize) -> Result<Bytes> {
        let index = self.index.read().unwrap();
        
        let entry = index.get(entry_id)
            .ok_or_else(|| StorageError::MemoryMap(
                format!("Entry {} not found", entry_id)
            ))?;
        
        self.storage.read_bytes(entry.offset, entry.len)
    }

    pub fn iter(&self) -> Result<Vec<(usize, Bytes)>> {
        let index = self.index.read().unwrap();
        let mut results = Vec::new();
        
        for (id, entry) in index.iter().enumerate() {
            let data = self.storage.read_bytes(entry.offset, entry.len)?;
            results.push((id, data));
        }
        
        Ok(results)
    }
    
    fn calculate_checksum(data: &[u8]) -> u32 {
        // Simple CRC32 checksum implementation
        const CRC32_TABLE: [u32; 256] = generate_crc32_table();
        
        let mut crc = 0xFFFFFFFFu32;
        for byte in data {
            let index = ((crc as u8) ^ byte) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[index];
        }
        
        !crc
    }
}

const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc = crc >> 1;
            }
            j += 1;
        }
        
        table[i] = crc;
        i += 1;
    }
    
    table
}

#[repr(C, packed)]
#[derive(IntoBytes, FromBytes, Immutable, Debug, Clone, Copy)]
struct EntryHeader {
    len: u32,
    timestamp: u64,
    checksum: u32,
}

// Memory-mapped ring buffer for streaming data
pub struct RingBuffer {
    storage: MmapStorage,
    capacity: usize,
    write_pos: Arc<RwLock<usize>>,
    read_pos: Arc<RwLock<usize>>,
}

impl RingBuffer {
    pub fn create(path: impl AsRef<Path>, capacity: usize) -> Result<Self> {
        let storage = MmapStorage::create(path, capacity + std::mem::size_of::<RingBufferHeader>())?;
        
        // Initialize header
        let header = RingBufferHeader {
            magic: RingBufferHeader::MAGIC,
            version: RingBufferHeader::VERSION,
            capacity: capacity as u64,
            write_pos: 0,
            read_pos: 0,
        };
        
        storage.write_bytes(0, header.as_bytes())?;
        
        Ok(Self {
            storage,
            capacity,
            write_pos: Arc::new(RwLock::new(0)),
            read_pos: Arc::new(RwLock::new(0)),
        })
    }

    pub fn write(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.capacity {
            return Err(StorageError::MemoryMap(
                "Data too large for ring buffer".to_string()
            ));
        }
        
        let mut write_pos = self.write_pos.write().unwrap();
        let header_size = std::mem::size_of::<RingBufferHeader>();
        
        // Calculate actual position in mmap
        let offset = header_size + *write_pos;
        
        // Handle wrap-around
        if *write_pos + data.len() > self.capacity {
            let first_part = self.capacity - *write_pos;
            self.storage.write_bytes(offset, &data[..first_part])?;
            self.storage.write_bytes(header_size, &data[first_part..])?;
            *write_pos = data.len() - first_part;
        } else {
            self.storage.write_bytes(offset, data)?;
            *write_pos = (*write_pos + data.len()) % self.capacity;
        }
        
        // Update header
        self.update_positions()?;
        
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        let mut read_pos = self.read_pos.write().unwrap();
        let write_pos = *self.write_pos.read().unwrap();
        
        if *read_pos == write_pos {
            return Ok(0); // Buffer empty
        }
        
        let header_size = std::mem::size_of::<RingBufferHeader>();
        let available = if write_pos >= *read_pos {
            write_pos - *read_pos
        } else {
            self.capacity - *read_pos + write_pos
        };
        
        let to_read = buf.len().min(available);
        
        // Read data
        if *read_pos + to_read > self.capacity {
            let first_part = self.capacity - *read_pos;
            let data1 = self.storage.read_bytes(header_size + *read_pos, first_part)?;
            let data2 = self.storage.read_bytes(header_size, to_read - first_part)?;
            
            buf[..first_part].copy_from_slice(&data1);
            buf[first_part..to_read].copy_from_slice(&data2);
            
            *read_pos = to_read - first_part;
        } else {
            let data = self.storage.read_bytes(header_size + *read_pos, to_read)?;
            buf[..to_read].copy_from_slice(&data);
            *read_pos = (*read_pos + to_read) % self.capacity;
        }
        
        // Update header
        self.update_positions()?;
        
        Ok(to_read)
    }

    fn update_positions(&self) -> Result<()> {
        let header = RingBufferHeader {
            magic: RingBufferHeader::MAGIC,
            version: RingBufferHeader::VERSION,
            capacity: self.capacity as u64,
            write_pos: *self.write_pos.read().unwrap() as u64,
            read_pos: *self.read_pos.read().unwrap() as u64,
        };
        
        self.storage.write_bytes(0, header.as_bytes())?;
        Ok(())
    }
}

#[repr(C, packed)]
#[derive(IntoBytes, FromBytes, Immutable, Debug, Clone, Copy)]
struct RingBufferHeader {
    magic: [u8; 8],
    version: u32,
    capacity: u64,
    write_pos: u64,
    read_pos: u64,
}

impl RingBufferHeader {
    const MAGIC: [u8; 8] = *b"MCPRRING";
    const VERSION: u32 = 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mmap");
        
        let storage = MmapStorage::create(&path, 1024).unwrap();
        
        // Test write and read
        let data = b"Hello, mmap!";
        storage.write_bytes(100, data).unwrap();
        
        let read_data = storage.read_bytes(100, data.len()).unwrap();
        assert_eq!(&read_data[..], data);
        
        // Test append
        let offset = storage.append(b"Appended data").unwrap();
        assert!(offset > 0);
    }

    #[test]
    fn test_append_log() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.log");
        
        let log = AppendLog::create(&path).unwrap();
        
        // Append entries
        let id1 = log.append(b"Entry 1").unwrap();
        let id2 = log.append(b"Entry 2").unwrap();
        
        // Read entries
        let data1 = log.read(id1).unwrap();
        let data2 = log.read(id2).unwrap();
        
        assert_eq!(&data1[..], b"Entry 1");
        assert_eq!(&data2[..], b"Entry 2");
    }

    #[test]
    fn test_ring_buffer() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ring");
        
        let ring = RingBuffer::create(&path, 100).unwrap();
        
        // Write data
        ring.write(b"Hello").unwrap();
        ring.write(b" World").unwrap();
        
        // Read data
        let mut buf = vec![0; 11];
        let n = ring.read(&mut buf).unwrap();
        
        assert_eq!(n, 11);
        assert_eq!(&buf[..n], b"Hello World");
    }
}