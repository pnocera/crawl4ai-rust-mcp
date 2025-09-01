use duckdb::{Connection, params, ToSql, Row};
use std::path::Path;
use std::sync::{Arc, Mutex};
use serde_json::Value;
use uuid::Uuid;

use crate::{Result, StorageError};

// DuckDB store for analytics and complex queries
pub struct DuckDbStore {
    conn: Arc<Mutex<Connection>>,
}

impl DuckDbStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        
        // Initialize enhanced schema
        conn.execute_batch(
            "-- Main pages table
            CREATE TABLE IF NOT EXISTS crawled_pages (
                id VARCHAR PRIMARY KEY,
                url VARCHAR NOT NULL,
                title VARCHAR,
                content TEXT,
                domain VARCHAR,
                crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_length INTEGER,
                content_hash VARCHAR,
                language VARCHAR,
                metadata JSON
            );
            
            -- Content chunks for RAG
            CREATE TABLE IF NOT EXISTS content_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                page_id VARCHAR NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_order INTEGER NOT NULL,
                chunk_size INTEGER NOT NULL,
                embedding_dims INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (page_id) REFERENCES crawled_pages(id)
            );
            
            -- Code examples extraction
            CREATE TABLE IF NOT EXISTS code_examples (
                example_id VARCHAR PRIMARY KEY,
                page_id VARCHAR NOT NULL,
                code_type VARCHAR NOT NULL,
                code_content TEXT NOT NULL,
                programming_language VARCHAR,
                context_text TEXT,
                line_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (page_id) REFERENCES crawled_pages(id)
            );
            
            -- Link graph for web crawling
            CREATE TABLE IF NOT EXISTS page_links (
                from_page_id VARCHAR NOT NULL,
                to_url VARCHAR NOT NULL,
                link_text VARCHAR,
                link_type VARCHAR,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_page_id) REFERENCES crawled_pages(id)
            );
            
            -- Search analytics
            CREATE TABLE IF NOT EXISTS search_queries (
                query_id VARCHAR PRIMARY KEY,
                query_text VARCHAR NOT NULL,
                result_count INTEGER NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                filters JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Performance indexes
            CREATE INDEX IF NOT EXISTS idx_pages_domain ON crawled_pages(domain);
            CREATE INDEX IF NOT EXISTS idx_pages_crawled_at ON crawled_pages(crawled_at);
            CREATE INDEX IF NOT EXISTS idx_pages_content_hash ON crawled_pages(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON content_chunks(page_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_order ON content_chunks(page_id, chunk_order);
            CREATE INDEX IF NOT EXISTS idx_code_page_id ON code_examples(page_id);
            CREATE INDEX IF NOT EXISTS idx_code_language ON code_examples(programming_language);
            CREATE INDEX IF NOT EXISTS idx_links_from ON page_links(from_page_id);
            CREATE INDEX IF NOT EXISTS idx_links_to ON page_links(to_url);
            CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON search_queries(timestamp);"
        )?;
        
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn insert_page(
        &self,
        id: &str,
        url: &str,
        title: &str,
        content: &str,
        domain: &str,
        language: Option<&str>,
        metadata: &Value,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let content_hash = format!("{:x}", md5::compute(content.as_bytes()));
        
        conn.execute(
            "INSERT OR REPLACE INTO crawled_pages 
             (id, url, title, content, domain, content_length, content_hash, language, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                id, url, title, content, domain,
                content.len() as i32,
                content_hash,
                language.unwrap_or("unknown"),
                metadata.to_string()
            ],
        )?;
        
        Ok(())
    }

    pub fn insert_content_chunk(
        &self,
        chunk_id: &str,
        page_id: &str,
        chunk_text: &str,
        chunk_order: i32,
        embedding_dims: Option<i32>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "INSERT INTO content_chunks (chunk_id, page_id, chunk_text, chunk_order, chunk_size, embedding_dims)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                chunk_id, page_id, chunk_text, chunk_order,
                chunk_text.len() as i32,
                embedding_dims
            ],
        )?;
        
        Ok(())
    }

    pub fn insert_code_example(
        &self,
        example_id: &str,
        page_id: &str,
        code_type: &str,
        code_content: &str,
        programming_language: Option<&str>,
        context_text: Option<&str>,
        line_number: Option<i32>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "INSERT INTO code_examples 
             (example_id, page_id, code_type, code_content, programming_language, context_text, line_number)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                example_id, page_id, code_type, code_content,
                programming_language, context_text, line_number
            ],
        )?;
        
        Ok(())
    }

    pub fn insert_page_link(
        &self,
        from_page_id: &str,
        to_url: &str,
        link_text: Option<&str>,
        link_type: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "INSERT INTO page_links (from_page_id, to_url, link_text, link_type)
             VALUES (?, ?, ?, ?)",
            params![from_page_id, to_url, link_text, link_type],
        )?;
        
        Ok(())
    }

    pub fn log_search_query(
        &self,
        query_id: &str,
        query_text: &str,
        result_count: i32,
        execution_time_ms: i32,
        filters: Option<&Value>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "INSERT INTO search_queries (query_id, query_text, result_count, execution_time_ms, filters)
             VALUES (?, ?, ?, ?, ?)",
            params![
                query_id, query_text, result_count, execution_time_ms,
                filters.map(|f| f.to_string())
            ],
        )?;
        
        Ok(())
    }

    // Enhanced query methods
    pub fn query_by_domain(&self, domain: &str, limit: Option<i32>) -> Result<Vec<PageSummary>> {
        let conn = self.conn.lock().unwrap();
        
        let query = format!(
            "SELECT id, url, title, crawled_at, content_length, language
             FROM crawled_pages 
             WHERE domain = ? 
             ORDER BY crawled_at DESC
             {}",
            limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
        );
        
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params![domain], |row| {
            Ok(PageSummary {
                id: row.get(0)?,
                url: row.get(1)?,
                title: row.get(2)?,
                crawled_at: row.get(3)?,
                content_length: row.get(4)?,
                language: row.get(5)?,
            })
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        
        Ok(results)
    }

    pub fn get_domain_statistics(&self) -> Result<Vec<DomainStats>> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT domain, 
                    COUNT(*) as page_count,
                    AVG(content_length) as avg_content_length,
                    MAX(crawled_at) as last_crawled,
                    COUNT(DISTINCT language) as language_count
             FROM crawled_pages 
             GROUP BY domain 
             ORDER BY page_count DESC"
        )?;
        
        let rows = stmt.query_map([], |row| {
            Ok(DomainStats {
                domain: row.get(0)?,
                page_count: row.get(1)?,
                avg_content_length: row.get(2)?,
                last_crawled: row.get(3)?,
                language_count: row.get(4)?,
            })
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        
        Ok(results)
    }

    pub fn search_code_examples(
        &self,
        programming_language: Option<&str>,
        keyword: Option<&str>,
        limit: Option<i32>,
    ) -> Result<Vec<CodeExample>> {
        let conn = self.conn.lock().unwrap();
        
        let mut conditions = Vec::new();
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();
        
        if let Some(lang) = programming_language {
            conditions.push("programming_language = ?");
            params.push(Box::new(lang.to_string()));
        }
        
        if let Some(kw) = keyword {
            conditions.push("(code_content LIKE ? OR context_text LIKE ?)");
            let pattern = format!("%{}%", kw);
            params.push(Box::new(pattern.clone()));
            params.push(Box::new(pattern));
        }
        
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };
        
        let query = format!(
            "SELECT ce.example_id, ce.page_id, ce.code_content, ce.programming_language,
                    ce.context_text, ce.line_number, cp.url, cp.title
             FROM code_examples ce
             JOIN crawled_pages cp ON ce.page_id = cp.id
             {}
             ORDER BY ce.created_at DESC
             {}",
            where_clause,
            limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
        );
        
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(
            params.iter().map(|p| p.as_ref()).collect::<Vec<_>>().as_slice(),
            |row| {
                Ok(CodeExample {
                    example_id: row.get(0)?,
                    page_id: row.get(1)?,
                    code_content: row.get(2)?,
                    programming_language: row.get(3)?,
                    context_text: row.get(4)?,
                    line_number: row.get(5)?,
                    page_url: row.get(6)?,
                    page_title: row.get(7)?,
                })
            }
        )?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        
        Ok(results)
    }

    pub fn get_crawl_timeline(&self, days: i32) -> Result<Vec<CrawlTimelineEntry>> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT DATE(crawled_at) as crawl_date,
                    COUNT(*) as pages_crawled,
                    COUNT(DISTINCT domain) as domains_crawled,
                    AVG(content_length) as avg_content_length
             FROM crawled_pages 
             WHERE crawled_at >= DATE('now', '-{} days')
             GROUP BY DATE(crawled_at)
             ORDER BY crawl_date DESC"
        )?;
        
        let rows = stmt.query_map(params![days], |row| {
            Ok(CrawlTimelineEntry {
                date: row.get(0)?,
                pages_crawled: row.get(1)?,
                domains_crawled: row.get(2)?,
                avg_content_length: row.get(3)?,
            })
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        
        Ok(results)
    }

    pub fn execute_custom_query(&self, query: &str) -> Result<Vec<Value>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(query)?;
        
        // Get column information
        let column_count = stmt.column_count();
        let column_names: Vec<String> = (0..column_count)
            .map(|i| stmt.column_name(i).map_or("unknown".to_string(), |s| s.to_string()))
            .collect();
        
        let rows = stmt.query_map([], |row| {
            let mut result = serde_json::Map::new();
            
            for (i, col_name) in column_names.iter().enumerate() {
                let value = match row.get::<_, Option<String>>(i) {
                    Ok(Some(s)) => Value::String(s),
                    Ok(None) => Value::Null,
                    Err(_) => {
                        // Try different types
                        if let Ok(i_val) = row.get::<_, i64>(i) {
                            Value::Number(serde_json::Number::from(i_val))
                        } else if let Ok(f_val) = row.get::<_, f64>(i) {
                            Value::Number(serde_json::Number::from_f64(f_val).unwrap_or_else(|| serde_json::Number::from(0)))
                        } else {
                            Value::Null
                        }
                    }
                };
                result.insert(col_name.clone(), value);
            }
            
            Ok(Value::Object(result))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        
        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct PageSummary {
    pub id: String,
    pub url: String,
    pub title: String,
    pub crawled_at: String,
    pub content_length: i32,
    pub language: String,
}

#[derive(Debug, Clone)]
pub struct DomainStats {
    pub domain: String,
    pub page_count: i32,
    pub avg_content_length: f64,
    pub last_crawled: String,
    pub language_count: i32,
}

#[derive(Debug, Clone)]
pub struct CodeExample {
    pub example_id: String,
    pub page_id: String,
    pub code_content: String,
    pub programming_language: Option<String>,
    pub context_text: Option<String>,
    pub line_number: Option<i32>,
    pub page_url: String,
    pub page_title: String,
}

#[derive(Debug, Clone)]
pub struct CrawlTimelineEntry {
    pub date: String,
    pub pages_crawled: i32,
    pub domains_crawled: i32,
    pub avg_content_length: f64,
}