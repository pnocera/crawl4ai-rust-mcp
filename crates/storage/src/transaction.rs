use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::{Result, StorageError, HybridStorage};

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Transaction states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Active,
    Committed,
    Aborted,
    Preparing,
}

/// Transaction operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOperation {
    Put { 
        store: String, 
        key: Vec<u8>, 
        value: Vec<u8>,
        old_value: Option<Vec<u8>>, // For rollback
    },
    Delete { 
        store: String, 
        key: Vec<u8>,
        old_value: Option<Vec<u8>>, // For rollback
    },
    BatchPut { 
        store: String, 
        items: Vec<(Vec<u8>, Vec<u8>)>,
        old_values: Vec<Option<Vec<u8>>>, // For rollback
    },
}

/// Write-ahead log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub transaction_id: Uuid,
    pub operation: TransactionOperation,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
}

/// Transaction context
#[derive(Debug)]
pub struct Transaction {
    pub id: Uuid,
    pub isolation_level: IsolationLevel,
    pub state: TransactionState,
    pub operations: Vec<TransactionOperation>,
    pub read_timestamp: DateTime<Utc>,
    pub write_timestamp: Option<DateTime<Utc>>,
}

impl Transaction {
    pub fn new(isolation_level: IsolationLevel) -> Self {
        Self {
            id: Uuid::new_v4(),
            isolation_level,
            state: TransactionState::Active,
            operations: Vec::new(),
            read_timestamp: Utc::now(),
            write_timestamp: None,
        }
    }

    pub fn add_operation(&mut self, operation: TransactionOperation) {
        self.operations.push(operation);
    }

    pub fn prepare(&mut self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(StorageError::TransactionError(
                format!("Cannot prepare transaction in state {:?}", self.state)
            ));
        }
        self.state = TransactionState::Preparing;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        if self.state != TransactionState::Preparing && self.state != TransactionState::Active {
            return Err(StorageError::TransactionError(
                format!("Cannot commit transaction in state {:?}", self.state)
            ));
        }
        self.state = TransactionState::Committed;
        self.write_timestamp = Some(Utc::now());
        Ok(())
    }

    pub fn abort(&mut self) -> Result<()> {
        if self.state == TransactionState::Committed {
            return Err(StorageError::TransactionError(
                "Cannot abort committed transaction".to_string()
            ));
        }
        self.state = TransactionState::Aborted;
        Ok(())
    }
}

/// Transaction manager with MVCC support
pub struct TransactionManager {
    storage: Arc<HybridStorage>,
    active_transactions: Arc<Mutex<HashMap<Uuid, Transaction>>>,
    transaction_log: Arc<Mutex<Vec<WalEntry>>>,
    sequence_counter: Arc<Mutex<u64>>,
}

impl TransactionManager {
    pub fn new(storage: Arc<HybridStorage>) -> Self {
        Self {
            storage,
            active_transactions: Arc::new(Mutex::new(HashMap::new())),
            transaction_log: Arc::new(Mutex::new(Vec::new())),
            sequence_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self, isolation_level: IsolationLevel) -> Result<Uuid> {
        let transaction = Transaction::new(isolation_level);
        let id = transaction.id;
        
        {
            let mut transactions = self.active_transactions.lock().unwrap();
            transactions.insert(id, transaction);
        }

        tracing::info!("Started transaction {} with isolation level {:?}", id, isolation_level);
        Ok(id)
    }

    /// Execute a transactional put operation
    pub fn transactional_put(
        &self,
        transaction_id: Uuid,
        store: &str,
        key: &[u8],
        value: &[u8],
    ) -> Result<()> {
        let mut transactions = self.active_transactions.lock().unwrap();
        let transaction = transactions.get_mut(&transaction_id)
            .ok_or_else(|| StorageError::TransactionNotFound(transaction_id))?;

        if transaction.state != TransactionState::Active {
            return Err(StorageError::TransactionError(
                format!("Transaction {} is not active", transaction_id)
            ));
        }

        // Get old value for rollback capability
        let old_value = match store {
            "metadata" => {
                self.storage.metadata_store()
                    .get(key)
                    .ok()
                    .flatten()
                    .and_then(|v: Vec<u8>| Some(v))
            },
            _ => None, // Add other stores as needed
        };

        let operation = TransactionOperation::Put {
            store: store.to_string(),
            key: key.to_vec(),
            value: value.to_vec(),
            old_value,
        };

        transaction.add_operation(operation.clone());
        
        // Write to WAL
        self.write_wal_entry(transaction_id, operation)?;

        Ok(())
    }

    /// Execute a transactional delete operation
    pub fn transactional_delete(
        &self,
        transaction_id: Uuid,
        store: &str,
        key: &[u8],
    ) -> Result<()> {
        let mut transactions = self.active_transactions.lock().unwrap();
        let transaction = transactions.get_mut(&transaction_id)
            .ok_or_else(|| StorageError::TransactionNotFound(transaction_id))?;

        if transaction.state != TransactionState::Active {
            return Err(StorageError::TransactionError(
                format!("Transaction {} is not active", transaction_id)
            ));
        }

        // Get old value for rollback capability
        let old_value = match store {
            "metadata" => {
                self.storage.metadata_store()
                    .get(key)
                    .ok()
                    .flatten()
                    .and_then(|v: Vec<u8>| Some(v))
            },
            _ => None, // Add other stores as needed
        };

        let operation = TransactionOperation::Delete {
            store: store.to_string(),
            key: key.to_vec(),
            old_value,
        };

        transaction.add_operation(operation.clone());
        
        // Write to WAL
        self.write_wal_entry(transaction_id, operation)?;

        Ok(())
    }

    /// Commit a transaction using two-phase commit
    pub async fn commit_transaction(&self, transaction_id: Uuid) -> Result<()> {
        // Phase 1: Prepare
        self.prepare_transaction(transaction_id)?;
        
        // Phase 2: Commit
        self.do_commit_transaction(transaction_id).await
    }

    /// Prepare phase of two-phase commit
    fn prepare_transaction(&self, transaction_id: Uuid) -> Result<()> {
        let mut transactions = self.active_transactions.lock().unwrap();
        let transaction = transactions.get_mut(&transaction_id)
            .ok_or_else(|| StorageError::TransactionNotFound(transaction_id))?;

        transaction.prepare()?;
        
        // Validate all operations can be applied
        for operation in &transaction.operations {
            self.validate_operation(operation)?;
        }

        tracing::info!("Transaction {} prepared successfully", transaction_id);
        Ok(())
    }

    /// Commit phase of two-phase commit
    async fn do_commit_transaction(&self, transaction_id: Uuid) -> Result<()> {
        let mut transactions = self.active_transactions.lock().unwrap();
        let mut transaction = transactions.remove(&transaction_id)
            .ok_or_else(|| StorageError::TransactionNotFound(transaction_id))?;

        // Apply all operations
        for operation in &transaction.operations {
            self.apply_operation(operation).await?;
        }

        transaction.commit()?;
        
        // Write commit record to WAL
        let commit_entry = WalEntry {
            transaction_id,
            operation: TransactionOperation::Put {
                store: "wal".to_string(),
                key: b"commit".to_vec(),
                value: transaction_id.as_bytes().to_vec(),
                old_value: None,
            },
            timestamp: Utc::now(),
            sequence_number: self.next_sequence_number(),
        };

        {
            let mut log = self.transaction_log.lock().unwrap();
            log.push(commit_entry);
        }

        tracing::info!("Transaction {} committed successfully", transaction_id);
        Ok(())
    }

    /// Rollback a transaction
    pub async fn rollback_transaction(&self, transaction_id: Uuid) -> Result<()> {
        let mut transactions = self.active_transactions.lock().unwrap();
        let mut transaction = transactions.remove(&transaction_id)
            .ok_or_else(|| StorageError::TransactionNotFound(transaction_id))?;

        // Apply rollback operations in reverse order
        for operation in transaction.operations.iter().rev() {
            self.rollback_operation(operation).await?;
        }

        transaction.abort()?;
        
        tracing::info!("Transaction {} rolled back successfully", transaction_id);
        Ok(())
    }

    /// Write an entry to the write-ahead log
    fn write_wal_entry(&self, transaction_id: Uuid, operation: TransactionOperation) -> Result<()> {
        let entry = WalEntry {
            transaction_id,
            operation,
            timestamp: Utc::now(),
            sequence_number: self.next_sequence_number(),
        };

        {
            let mut log = self.transaction_log.lock().unwrap();
            log.push(entry);
        }

        Ok(())
    }

    /// Get next sequence number
    fn next_sequence_number(&self) -> u64 {
        let mut counter = self.sequence_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Validate an operation can be applied
    fn validate_operation(&self, _operation: &TransactionOperation) -> Result<()> {
        // Basic validation - can be extended for conflict detection
        Ok(())
    }

    /// Apply an operation to storage
    async fn apply_operation(&self, operation: &TransactionOperation) -> Result<()> {
        match operation {
            TransactionOperation::Put { store, key, value, .. } => {
                match store.as_str() {
                    "metadata" => {
                        self.storage.metadata_store()
                            .put(key, &value.clone())?;
                    },
                    _ => {
                        return Err(StorageError::Other(
                            format!("Unknown store: {}", store)
                        ));
                    }
                }
            },
            TransactionOperation::Delete { store, key, .. } => {
                match store.as_str() {
                    "metadata" => {
                        self.storage.metadata_store()
                            .delete(key)?;
                    },
                    _ => {
                        return Err(StorageError::Other(
                            format!("Unknown store: {}", store)
                        ));
                    }
                }
            },
            TransactionOperation::BatchPut { store, items, .. } => {
                match store.as_str() {
                    "metadata" => {
                        let byte_items: Vec<(Vec<u8>, Vec<u8>)> = items.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        self.storage.metadata_store()
                            .batch_put_cf::<Vec<u8>, Vec<u8>>("default", byte_items)?;
                    },
                    _ => {
                        return Err(StorageError::Other(
                            format!("Unknown store: {}", store)
                        ));
                    }
                }
            },
        }
        Ok(())
    }

    /// Rollback an operation
    async fn rollback_operation(&self, operation: &TransactionOperation) -> Result<()> {
        match operation {
            TransactionOperation::Put { store, key, old_value, .. } => {
                match store.as_str() {
                    "metadata" => {
                        if let Some(old_val) = old_value {
                            self.storage.metadata_store()
                                .put(key, old_val)?;
                        } else {
                            self.storage.metadata_store()
                                .delete(key)?;
                        }
                    },
                    _ => {
                        return Err(StorageError::Other(
                            format!("Unknown store: {}", store)
                        ));
                    }
                }
            },
            TransactionOperation::Delete { store, key, old_value, .. } => {
                if let Some(old_val) = old_value {
                    match store.as_str() {
                        "metadata" => {
                            self.storage.metadata_store()
                                .put(key, old_val)?;
                        },
                        _ => {
                            return Err(StorageError::Other(
                                format!("Unknown store: {}", store).into()
                            ));
                        }
                    }
                }
            },
            TransactionOperation::BatchPut { store, items, old_values } => {
                match store.as_str() {
                    "metadata" => {
                        for (i, (key, _)) in items.iter().enumerate() {
                            if let Some(Some(old_val)) = old_values.get(i) {
                                self.storage.metadata_store()
                                    .put(key, old_val)?;
                            } else {
                                self.storage.metadata_store()
                                    .delete(key)?;
                            }
                        }
                    },
                    _ => {
                        return Err(StorageError::Other(
                            format!("Unknown store: {}", store)
                        ));
                    }
                }
            },
        }
        Ok(())
    }

    /// Get all active transactions
    pub fn get_active_transactions(&self) -> Vec<Uuid> {
        let transactions = self.active_transactions.lock().unwrap();
        transactions.keys().cloned().collect()
    }

    /// Get transaction status
    pub fn get_transaction_state(&self, transaction_id: Uuid) -> Option<TransactionState> {
        let transactions = self.active_transactions.lock().unwrap();
        transactions.get(&transaction_id).map(|t| t.state)
    }
}