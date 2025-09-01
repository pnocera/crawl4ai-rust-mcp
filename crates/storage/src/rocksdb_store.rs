use rocksdb::{DB, Options, WriteBatch, ColumnFamily, ColumnFamilyDescriptor, MergeOperands, BlockBasedOptions, Cache};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::{Result, StorageError};

// Column family names
pub const CF_METADATA: &str = "metadata";
pub const CF_INDEXES: &str = "indexes";
pub const CF_COUNTERS: &str = "counters";
pub const CF_CACHE: &str = "cache";

// Enhanced RocksDB store with multiple column families and advanced features
pub struct RocksDbStore {
    db: Arc<DB>,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    cache_size: usize,
}

#[derive(Clone)]
struct CacheEntry {
    data: Vec<u8>,
    timestamp: u64,
    access_count: u64,
}

impl RocksDbStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        
        // Performance optimizations
        opts.set_max_background_jobs(4);
        opts.set_bytes_per_sync(1024 * 1024); // 1MB
        opts.set_wal_bytes_per_sync(1024 * 1024);
        opts.set_max_write_buffer_number(4);
        opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB
        
        // Block cache for better read performance
        let cache = Cache::new_lru_cache(256 * 1024 * 1024); // 256MB
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        block_opts.set_block_size(16 * 1024); // 16KB
        opts.set_block_based_table_factory(&block_opts);
        
        // Set up merge operator for counters
        opts.set_merge_operator_associative("increment", increment_merge);
        
        // Column families
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_METADATA, Options::default()),
            ColumnFamilyDescriptor::new(CF_INDEXES, Options::default()),
            ColumnFamilyDescriptor::new(CF_COUNTERS, {
                let mut cf_opts = Options::default();
                cf_opts.set_merge_operator_associative("increment", increment_merge);
                cf_opts
            }),
            ColumnFamilyDescriptor::new(CF_CACHE, Options::default()),
        ];
        
        let db = DB::open_cf_descriptors(&opts, path_ref, cf_descriptors)?;
        
        info!("RocksDB store initialized at {:?}", path_ref);
        
        Ok(Self {
            db: Arc::new(db),
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_size: 10000, // Max cache entries
        })
    }

    pub fn put<K, V>(&self, key: K, value: &V) -> Result<()>
    where
        K: AsRef<[u8]>,
        V: Serialize,
    {
        let encoded = bincode::serialize(value)?;
        self.db.put(key, encoded)?;
        Ok(())
    }

    pub fn get<K, V>(&self, key: K) -> Result<Option<V>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        match self.db.get(key)? {
            Some(data) => {
                let value = bincode::deserialize(&data)?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    pub fn delete<K>(&self, key: K) -> Result<()>
    where
        K: AsRef<[u8]>,
    {
        self.db.delete(key)?;
        Ok(())
    }

    pub fn batch_write(&self, batch: WriteBatch) -> Result<()> {
        self.db.write(batch)?;
        Ok(())
    }

    pub fn prefix_iterator(&self, prefix: &[u8]) -> impl Iterator<Item = (Box<[u8]>, Box<[u8]>)> + '_ {
        self.db
            .prefix_iterator(prefix)
            .map(|item| item.unwrap())
    }

    // Enhanced methods with column family support
    pub fn put_cf<K, V>(&self, cf_name: &str, key: K, value: &V) -> Result<()>
    where
        K: AsRef<[u8]>,
        V: Serialize,
    {
        let cf = self.db.cf_handle(cf_name)
            .ok_or_else(|| StorageError::NotFound(format!("Column family {} not found", cf_name)))?;
        let encoded = bincode::serialize(value)?;
        self.db.put_cf(&cf, key, encoded)?;
        Ok(())
    }

    pub fn get_cf<K, V>(&self, cf_name: &str, key: K) -> Result<Option<V>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        let cf = self.db.cf_handle(cf_name)
            .ok_or_else(|| StorageError::NotFound(format!("Column family {} not found", cf_name)))?;
        
        match self.db.get_cf(&cf, key)? {
            Some(data) => {
                let value = bincode::deserialize(&data)?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    pub fn delete_cf<K>(&self, cf_name: &str, key: K) -> Result<()>
    where
        K: AsRef<[u8]>,
    {
        let cf = self.db.cf_handle(cf_name)
            .ok_or_else(|| StorageError::NotFound(format!("Column family {} not found", cf_name)))?;
        self.db.delete_cf(&cf, key)?;
        Ok(())
    }

    // Counter operations with merge
    pub fn increment_counter(&self, key: &str, amount: u64) -> Result<()> {
        let cf = self.db.cf_handle(CF_COUNTERS)
            .ok_or_else(|| StorageError::NotFound("Counters column family not found".to_string()))?;
        self.db.merge_cf(&cf, key, &amount.to_le_bytes())?;
        Ok(())
    }

    pub fn get_counter(&self, key: &str) -> Result<u64> {
        let cf = self.db.cf_handle(CF_COUNTERS)
            .ok_or_else(|| StorageError::NotFound("Counters column family not found".to_string()))?;
        
        match self.db.get_cf(&cf, key)? {
            Some(data) => {
                let bytes = data.as_slice().try_into()
                    .map_err(|_| StorageError::Other("Invalid counter data".to_string()))?;
                Ok(u64::from_le_bytes(bytes))
            }
            None => Ok(0),
        }
    }

    // Advanced batch operations
    pub fn batch_put_cf<K, V>(&self, cf_name: &str, items: Vec<(K, Vec<u8>)>) -> Result<()>
    where
        K: AsRef<[u8]>,
    {
        let cf = self.db.cf_handle(cf_name)
            .ok_or_else(|| StorageError::NotFound(format!("Column family {} not found", cf_name)))?;
        
        let mut batch = WriteBatch::default();
        for (key, value) in items {
            batch.put_cf(&cf, key, value);
        }
        
        self.db.write(batch)?;
        Ok(())
    }

    // Caching layer
    pub fn get_cached<K, V>(&self, key: K) -> Result<Option<V>>
    where
        K: AsRef<[u8]> + Clone,
        V: for<'de> Deserialize<'de> + Clone + 'static + Serialize,
    {
        let key_str = String::from_utf8_lossy(key.as_ref()).to_string();
        
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(&key_str) {
                if let Ok(value) = bincode::deserialize(&entry.data) {
                    // Update access count
                    drop(cache);
                    let mut cache_mut = self.cache.write().unwrap();
                    if let Some(entry) = cache_mut.get_mut(&key_str) {
                        entry.access_count += 1;
                    }
                    return Ok(Some(value));
                }
            }
        }
        
        // Cache miss - get from DB
        let result = self.get(key.clone())?;
        
        // Cache the result if found
        if let Some(ref value) = result {
            if let Ok(encoded) = bincode::serialize(value) {
                let mut cache = self.cache.write().unwrap();
                
                // Evict oldest if cache is full
                if cache.len() >= self.cache_size {
                    let oldest_key = cache.iter()
                        .min_by_key(|(_, entry)| entry.timestamp)
                        .map(|(k, _)| k.clone());
                    
                    if let Some(key) = oldest_key {
                        cache.remove(&key);
                    }
                }
                
                cache.insert(key_str, CacheEntry {
                    data: encoded,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    access_count: 1,
                });
            }
        }
        
        Ok(result)
    }

    pub fn invalidate_cache(&self, key: &str) {
        let mut cache = self.cache.write().unwrap();
        cache.remove(key);
    }

    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    // Statistics and monitoring
    pub fn get_stats(&self) -> Result<RocksDbStats> {
        let cache = self.cache.read().unwrap();
        let cache_size = cache.len();
        let total_access_count: u64 = cache.values().map(|e| e.access_count).sum();
        drop(cache);

        Ok(RocksDbStats {
            cache_size,
            cache_access_count: total_access_count,
            approximate_num_keys: self.db.property_int_value("rocksdb.estimate-num-keys")?.unwrap_or(0),
            total_sst_files_size: self.db.property_int_value("rocksdb.total-sst-files-size")?.unwrap_or(0),
            live_sst_files_size: self.db.property_int_value("rocksdb.live-sst-files-size")?.unwrap_or(0),
        })
    }
}

// Merge operator for counters
fn increment_merge(
    _new_key: &[u8],
    existing_val: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut counter = existing_val
        .and_then(|bytes| {
            if bytes.len() == 8 {
                Some(u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                    bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            } else {
                None
            }
        })
        .unwrap_or(0);
    
    for op in operands {
        if op.len() == 8 {
            let increment = u64::from_le_bytes([
                op[0], op[1], op[2], op[3],
                op[4], op[5], op[6], op[7],
            ]);
            counter += increment;
        }
    }
    
    Some(counter.to_le_bytes().to_vec())
}

#[derive(Debug, Clone)]
pub struct RocksDbStats {
    pub cache_size: usize,
    pub cache_access_count: u64,
    pub approximate_num_keys: u64,
    pub total_sst_files_size: u64,
    pub live_sst_files_size: u64,
}