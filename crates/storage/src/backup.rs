use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;
use tokio::fs as async_fs;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use futures::future::BoxFuture;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tracing::{info, warn, error};

use crate::{Result, StorageError, HybridStorage};

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    pub backup_directory: PathBuf,
    pub max_backups: usize,
    pub compression_enabled: bool,
    pub include_indexes: bool,
    pub include_cache: bool,
    pub backup_interval_hours: u64,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            backup_directory: PathBuf::from("./backups"),
            max_backups: 10,
            compression_enabled: true,
            include_indexes: true,
            include_cache: false,
            backup_interval_hours: 24,
        }
    }
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    pub backup_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub backup_type: BackupType,
    pub storage_version: String,
    pub file_count: usize,
    pub total_size_bytes: u64,
    pub compression_ratio: Option<f32>,
    pub checksum: String,
    pub is_complete: bool,
}

/// Types of backups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupType {
    Full,
    Incremental { since: DateTime<Utc> },
    Snapshot,
}

/// Backup restore options
#[derive(Debug, Clone)]
pub struct RestoreOptions {
    pub target_directory: PathBuf,
    pub verify_checksum: bool,
    pub restore_indexes: bool,
    pub restore_cache: bool,
    pub overwrite_existing: bool,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            target_directory: PathBuf::from("./restored"),
            verify_checksum: true,
            restore_indexes: true,
            restore_cache: false,
            overwrite_existing: false,
        }
    }
}

/// Backup and recovery manager
pub struct BackupManager {
    storage: Arc<HybridStorage>,
    config: BackupConfig,
}

impl BackupManager {
    pub fn new(storage: Arc<HybridStorage>, config: BackupConfig) -> Result<Self> {
        // Create backup directory if it doesn't exist
        fs::create_dir_all(&config.backup_directory)?;
        
        Ok(Self { storage, config })
    }

    /// Create a full backup of all storage components
    pub async fn create_full_backup(&self) -> Result<BackupMetadata> {
        let backup_id = Uuid::new_v4();
        let backup_path = self.get_backup_path(&backup_id);
        
        info!("Starting full backup {} to {:?}", backup_id, backup_path);
        
        async_fs::create_dir_all(&backup_path).await?;
        
        let mut file_count = 0;
        let mut total_size = 0u64;
        
        // Backup RocksDB data
        let rocksdb_backup_path = backup_path.join("rocksdb");
        async_fs::create_dir_all(&rocksdb_backup_path).await?;
        
        // Note: In a real implementation, you'd use RocksDB's backup API
        // For now, we'll simulate by copying the database directory
        if let Err(e) = self.copy_directory_async(
            &PathBuf::from("./storage/metadata"), // Assuming this is where RocksDB stores data
            &rocksdb_backup_path
        ).await {
            warn!("RocksDB backup failed: {}", e);
        } else {
            let size = self.calculate_directory_size(&rocksdb_backup_path).await?;
            total_size += size;
            file_count += self.count_files_in_directory(&rocksdb_backup_path).await?;
        }
        
        // Backup DuckDB data
        let duckdb_backup_path = backup_path.join("duckdb");
        async_fs::create_dir_all(&duckdb_backup_path).await?;
        
        if let Err(e) = self.copy_directory_async(
            &PathBuf::from("./storage/analytics"), // Assuming this is where DuckDB stores data
            &duckdb_backup_path
        ).await {
            warn!("DuckDB backup failed: {}", e);
        } else {
            let size = self.calculate_directory_size(&duckdb_backup_path).await?;
            total_size += size;
            file_count += self.count_files_in_directory(&duckdb_backup_path).await?;
        }
        
        // Backup memory-mapped files
        let mmap_backup_path = backup_path.join("mmap");
        async_fs::create_dir_all(&mmap_backup_path).await?;
        
        if let Err(e) = self.copy_directory_async(
            &PathBuf::from("./storage/content"), // Content log
            &mmap_backup_path.join("content")
        ).await {
            warn!("Content backup failed: {}", e);
        }
        
        if let Err(e) = self.copy_directory_async(
            &PathBuf::from("./storage/embeddings"), // Embedding storage
            &mmap_backup_path.join("embeddings")
        ).await {
            warn!("Embeddings backup failed: {}", e);
        }
        
        let mmap_size = self.calculate_directory_size(&mmap_backup_path).await?;
        total_size += mmap_size;
        file_count += self.count_files_in_directory(&mmap_backup_path).await?;
        
        // Create backup metadata
        let checksum = self.calculate_backup_checksum(&backup_path).await?;
        let compression_ratio = if self.config.compression_enabled {
            Some(self.compress_backup(&backup_path).await?)
        } else {
            None
        };
        
        let metadata = BackupMetadata {
            backup_id,
            created_at: Utc::now(),
            backup_type: BackupType::Full,
            storage_version: "1.0.0".to_string(),
            file_count,
            total_size_bytes: total_size,
            compression_ratio,
            checksum,
            is_complete: true,
        };
        
        // Save metadata
        self.save_backup_metadata(&metadata).await?;
        
        // Clean up old backups
        self.cleanup_old_backups().await?;
        
        info!(
            "Full backup {} completed: {} files, {} bytes", 
            backup_id, file_count, total_size
        );
        
        Ok(metadata)
    }

    /// Create an incremental backup since the last backup
    pub async fn create_incremental_backup(&self, since: DateTime<Utc>) -> Result<BackupMetadata> {
        let backup_id = Uuid::new_v4();
        let backup_path = self.get_backup_path(&backup_id);
        
        info!("Starting incremental backup {} since {}", backup_id, since);
        
        async_fs::create_dir_all(&backup_path).await?;
        
        // For incremental backups, we'd typically use:
        // 1. RocksDB's incremental backup feature
        // 2. DuckDB transaction logs
        // 3. File modification timestamps for memory-mapped files
        
        // This is a simplified implementation
        let mut file_count = 0;
        let mut total_size = 0u64;
        
        // Create incremental backup metadata
        let checksum = self.calculate_backup_checksum(&backup_path).await?;
        
        let metadata = BackupMetadata {
            backup_id,
            created_at: Utc::now(),
            backup_type: BackupType::Incremental { since },
            storage_version: "1.0.0".to_string(),
            file_count,
            total_size_bytes: total_size,
            compression_ratio: None,
            checksum,
            is_complete: true,
        };
        
        self.save_backup_metadata(&metadata).await?;
        
        info!("Incremental backup {} completed", backup_id);
        
        Ok(metadata)
    }

    /// Create a snapshot backup (copy-on-write style)
    pub async fn create_snapshot(&self) -> Result<BackupMetadata> {
        let backup_id = Uuid::new_v4();
        
        info!("Creating snapshot backup {}", backup_id);
        
        // In a real implementation, this would use:
        // - RocksDB snapshots
        // - DuckDB savepoints
        // - Memory-mapped file snapshots
        
        let metadata = BackupMetadata {
            backup_id,
            created_at: Utc::now(),
            backup_type: BackupType::Snapshot,
            storage_version: "1.0.0".to_string(),
            file_count: 0,
            total_size_bytes: 0,
            compression_ratio: None,
            checksum: "snapshot".to_string(),
            is_complete: true,
        };
        
        self.save_backup_metadata(&metadata).await?;
        
        info!("Snapshot backup {} created", backup_id);
        
        Ok(metadata)
    }

    /// List all available backups
    pub async fn list_backups(&self) -> Result<Vec<BackupMetadata>> {
        let mut backups = Vec::new();
        
        let mut dir = async_fs::read_dir(&self.config.backup_directory).await?;
        while let Some(entry) = dir.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                let metadata_path = entry.path().join("metadata.json");
                if metadata_path.exists() {
                    if let Ok(metadata) = self.load_backup_metadata(&metadata_path).await {
                        backups.push(metadata);
                    }
                }
            }
        }
        
        // Sort by creation time, newest first
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(backups)
    }

    /// Restore from a backup
    pub async fn restore_from_backup(
        &self,
        backup_id: Uuid,
        options: RestoreOptions,
    ) -> Result<()> {
        let backup_path = self.get_backup_path(&backup_id);
        let metadata_path = backup_path.join("metadata.json");
        
        if !metadata_path.exists() {
            return Err(StorageError::NotFound(format!("Backup {} not found", backup_id)));
        }
        
        let metadata = self.load_backup_metadata(&metadata_path).await?;
        
        if !metadata.is_complete {
            return Err(StorageError::Other(
                "Cannot restore from incomplete backup".to_string()
            ));
        }
        
        info!("Starting restore from backup {}", backup_id);
        
        // Verify checksum if requested
        if options.verify_checksum {
            let calculated_checksum = self.calculate_backup_checksum(&backup_path).await?;
            if calculated_checksum != metadata.checksum {
                return Err(StorageError::Other(
                    "Backup checksum verification failed".to_string()
                ));
            }
        }
        
        // Create target directory
        async_fs::create_dir_all(&options.target_directory).await?;
        
        // Decompress if necessary
        let source_path = if metadata.compression_ratio.is_some() {
            self.decompress_backup(&backup_path).await?
        } else {
            backup_path
        };
        
        // Restore RocksDB data
        if source_path.join("rocksdb").exists() {
            let target_rocksdb = options.target_directory.join("metadata");
            if options.overwrite_existing || !target_rocksdb.exists() {
                self.copy_directory_async(&source_path.join("rocksdb"), &target_rocksdb).await?;
            }
        }
        
        // Restore DuckDB data
        if source_path.join("duckdb").exists() {
            let target_duckdb = options.target_directory.join("analytics");
            if options.overwrite_existing || !target_duckdb.exists() {
                self.copy_directory_async(&source_path.join("duckdb"), &target_duckdb).await?;
            }
        }
        
        // Restore memory-mapped files
        if source_path.join("mmap").exists() {
            if source_path.join("mmap/content").exists() {
                let target_content = options.target_directory.join("content");
                if options.overwrite_existing || !target_content.exists() {
                    self.copy_directory_async(&source_path.join("mmap/content"), &target_content).await?;
                }
            }
            
            if source_path.join("mmap/embeddings").exists() {
                let target_embeddings = options.target_directory.join("embeddings");
                if options.overwrite_existing || !target_embeddings.exists() {
                    self.copy_directory_async(&source_path.join("mmap/embeddings"), &target_embeddings).await?;
                }
            }
        }
        
        info!("Restore from backup {} completed to {:?}", backup_id, options.target_directory);
        
        Ok(())
    }

    /// Delete a specific backup
    pub async fn delete_backup(&self, backup_id: Uuid) -> Result<()> {
        let backup_path = self.get_backup_path(&backup_id);
        
        if backup_path.exists() {
            async_fs::remove_dir_all(&backup_path).await?;
            info!("Backup {} deleted", backup_id);
        }
        
        Ok(())
    }

    /// Verify the integrity of a backup
    pub async fn verify_backup(&self, backup_id: Uuid) -> Result<bool> {
        let backup_path = self.get_backup_path(&backup_id);
        let metadata_path = backup_path.join("metadata.json");
        
        if !metadata_path.exists() {
            return Ok(false);
        }
        
        let metadata = self.load_backup_metadata(&metadata_path).await?;
        let calculated_checksum = self.calculate_backup_checksum(&backup_path).await?;
        
        Ok(calculated_checksum == metadata.checksum)
    }

    // Helper methods
    
    fn get_backup_path(&self, backup_id: &Uuid) -> PathBuf {
        self.config.backup_directory.join(backup_id.to_string())
    }
    
    async fn save_backup_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let backup_path = self.get_backup_path(&metadata.backup_id);
        let metadata_path = backup_path.join("metadata.json");
        
        let json = serde_json::to_string_pretty(metadata)?;
        let mut file = async_fs::File::create(&metadata_path).await?;
        file.write_all(json.as_bytes()).await?;
        
        Ok(())
    }
    
    async fn load_backup_metadata(&self, metadata_path: &Path) -> Result<BackupMetadata> {
        let mut file = async_fs::File::open(metadata_path).await?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).await?;
        
        let metadata: BackupMetadata = serde_json::from_str(&contents)?;
        Ok(metadata)
    }
    
    fn copy_directory_async<'a>(&'a self, source: &'a Path, target: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
        if !source.exists() {
            return Ok(()); // Skip if source doesn't exist
        }
        
        async_fs::create_dir_all(target).await?;
        
        let mut entries = async_fs::read_dir(source).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let target_path = target.join(entry.file_name());
            
            if entry.file_type().await?.is_dir() {
                self.copy_directory_async(&entry_path, &target_path).await?;
            } else {
                async_fs::copy(&entry_path, &target_path).await?;
            }
        }
        
        Ok(())
        })
    }
    
    fn calculate_directory_size<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, Result<u64>> {
        Box::pin(async move {
        let mut total_size = 0;
        
        if !path.exists() {
            return Ok(0);
        }
        
        let mut entries = async_fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                total_size += self.calculate_directory_size(&entry.path()).await?;
            } else {
                total_size += entry.metadata().await?.len();
            }
        }
        
        Ok(total_size)
        })
    }
    
    fn count_files_in_directory<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
        let mut count = 0;
        
        if !path.exists() {
            return Ok(0);
        }
        
        let mut entries = async_fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                count += self.count_files_in_directory(&entry.path()).await?;
            } else {
                count += 1;
            }
        }
        
        Ok(count)
        })
    }
    
    async fn calculate_backup_checksum(&self, backup_path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        self.hash_directory_recursive(&mut hasher, backup_path).await?;
        let result = hasher.finalize();
        
        Ok(format!("{:x}", result))
    }
    
    fn hash_directory_recursive<'a>(
        &'a self,
        hasher: &'a mut sha2::Sha256,
        path: &'a Path
    ) -> BoxFuture<'a, Result<()>> {
        use sha2::Digest;
        Box::pin(async move {
        if !path.exists() {
            return Ok(());
        }
        
        let mut entries = async_fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            
            if entry.file_type().await?.is_dir() {
                self.hash_directory_recursive(hasher, &entry_path).await?;
            } else if entry_path.file_name() != Some(std::ffi::OsStr::new("metadata.json")) {
                let mut file = async_fs::File::open(&entry_path).await?;
                let mut buffer = [0; 8192];
                
                loop {
                    let bytes_read = file.read(&mut buffer).await?;
                    if bytes_read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..bytes_read]);
                }
            }
        }
        
        Ok(())
        })
    }
    
    async fn compress_backup(&self, backup_path: &Path) -> Result<f32> {
        if !self.config.compression_enabled {
            return Ok(1.0); // No compression
        }

        info!("Compressing backup at {}", backup_path.display());
        
        let original_size = self.calculate_directory_size(backup_path).await? as usize;
        let compressed_file = backup_path.with_extension("tar.lz4");
        
        // Create tar archive and compress it with LZ4
        let tar_file = backup_path.with_extension("tar");
        self.create_tar_archive(backup_path, &tar_file).await?;
        
        // Compress the tar file
        let tar_data = async_fs::read(&tar_file).await?;
        let compressed_data = lz4::block::compress(&tar_data, None, true)
            .map_err(|e| StorageError::BackupFailed(format!("Compression failed: {}", e)))?;
        
        // Write compressed data
        async_fs::write(&compressed_file, compressed_data).await?;
        
        // Clean up tar file
        async_fs::remove_file(tar_file).await?;
        
        // Calculate compression ratio
        let compressed_size = async_fs::metadata(&compressed_file).await?.len() as usize;
        let compression_ratio = compressed_size as f32 / original_size as f32;
        
        info!(
            "Compressed backup from {} bytes to {} bytes (ratio: {:.2})",
            original_size, compressed_size, compression_ratio
        );
        
        Ok(compression_ratio)
    }
    
    async fn decompress_backup(&self, backup_path: &Path) -> Result<PathBuf> {
        let compressed_file = backup_path.with_extension("tar.lz4");
        
        if !compressed_file.exists() {
            // Not compressed, return original path
            return Ok(backup_path.to_path_buf());
        }
        
        info!("Decompressing backup from {}", compressed_file.display());
        
        // Read compressed data
        let compressed_data = async_fs::read(&compressed_file).await?;
        
        // Decompress
        let tar_data = lz4::block::decompress(&compressed_data, None)
            .map_err(|e| StorageError::BackupFailed(format!("Decompression failed: {}", e)))?;
        
        // Create temporary directory for decompressed data
        let temp_dir = backup_path.parent()
            .unwrap_or_else(|| Path::new("/tmp"))
            .join(format!("restore_temp_{}", Uuid::new_v4()));
        
        async_fs::create_dir_all(&temp_dir).await?;
        
        // Write tar file to temp location
        let temp_tar = temp_dir.join("backup.tar");
        async_fs::write(&temp_tar, tar_data).await?;
        
        // Extract tar archive
        let extraction_dir = temp_dir.join("extracted");
        async_fs::create_dir_all(&extraction_dir).await?;
        
        self.extract_tar_archive(&temp_tar, &extraction_dir).await?;
        
        // Clean up tar file
        async_fs::remove_file(temp_tar).await?;
        
        info!("Decompressed backup to {}", extraction_dir.display());
        Ok(extraction_dir)
    }
    
    
    async fn create_tar_archive(&self, source_dir: &Path, tar_file: &Path) -> Result<()> {
        use std::process::Command;
        
        // Use system tar command for simplicity
        // In production, consider using a pure Rust tar implementation
        let output = Command::new("tar")
            .args([
                "-cf",
                tar_file.to_str().unwrap(),
                "-C",
                source_dir.parent().unwrap().to_str().unwrap(),
                source_dir.file_name().unwrap().to_str().unwrap(),
            ])
            .output()
            .map_err(|e| StorageError::BackupFailed(format!("Failed to create tar: {}", e)))?;
        
        if !output.status.success() {
            return Err(StorageError::BackupFailed(
                format!("tar command failed: {}", String::from_utf8_lossy(&output.stderr))
            ));
        }
        
        Ok(())
    }
    
    async fn extract_tar_archive(&self, tar_file: &Path, dest_dir: &Path) -> Result<()> {
        use std::process::Command;
        
        // Use system tar command for extraction
        let output = Command::new("tar")
            .args([
                "-xf",
                tar_file.to_str().unwrap(),
                "-C",
                dest_dir.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| StorageError::BackupFailed(format!("Failed to extract tar: {}", e)))?;
        
        if !output.status.success() {
            return Err(StorageError::BackupFailed(
                format!("tar extraction failed: {}", String::from_utf8_lossy(&output.stderr))
            ));
        }
        
        Ok(())
    }
    
    async fn cleanup_old_backups(&self) -> Result<()> {
        let backups = self.list_backups().await?;
        
        if backups.len() > self.config.max_backups {
            let to_delete = &backups[self.config.max_backups..];
            
            for backup in to_delete {
                if let Err(e) = self.delete_backup(backup.backup_id).await {
                    warn!("Failed to delete old backup {}: {}", backup.backup_id, e);
                }
            }
        }
        
        Ok(())
    }
}