use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use neo4rs::{Graph, query, ConfigBuilder};
use tokio::fs;
use tracing::{debug, info, warn};
use walkdir::WalkDir;
use syn::{self, visit::Visit, Item};
use regex::Regex;

use crate::{GraphStoreError, MemgraphConfig};

pub struct GraphStore {
    graph: Graph,
    config: MemgraphConfig,
}

#[derive(Debug, Clone)]
pub struct ParseRepositoryResult {
    pub nodes_created: usize,
    pub relationships_created: usize,
    pub files_processed: usize,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f32,
    pub issues: Vec<ValidationIssue>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub line: u32,
    pub column: u32,
    pub severity: String,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
struct CodeElement {
    name: String,
    element_type: String,
    file_path: String,
    line_number: u32,
    parent: Option<String>,
    visibility: String,
    attributes: HashMap<String, String>,
}

struct AstVisitor {
    file_path: String,
    elements: Vec<CodeElement>,
    current_mod: Vec<String>,
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_item(&mut self, item: &'ast Item) {
        match item {
            Item::Fn(func) => {
                self.elements.push(CodeElement {
                    name: func.sig.ident.to_string(),
                    element_type: "function".to_string(),
                    file_path: self.file_path.clone(),
                    line_number: 1, // syn doesn't provide line numbers by default
                    parent: self.current_mod.last().cloned(),
                    visibility: self.format_visibility(&func.vis),
                    attributes: self.extract_attributes(&func.attrs),
                });
            }
            Item::Struct(s) => {
                self.elements.push(CodeElement {
                    name: s.ident.to_string(),
                    element_type: "struct".to_string(),
                    file_path: self.file_path.clone(),
                    line_number: 1,
                    parent: self.current_mod.last().cloned(),
                    visibility: self.format_visibility(&s.vis),
                    attributes: HashMap::new(),
                });
            }
            Item::Enum(e) => {
                self.elements.push(CodeElement {
                    name: e.ident.to_string(),
                    element_type: "enum".to_string(),
                    file_path: self.file_path.clone(),
                    line_number: 1,
                    parent: self.current_mod.last().cloned(),
                    visibility: self.format_visibility(&e.vis),
                    attributes: HashMap::new(),
                });
            }
            Item::Trait(t) => {
                self.elements.push(CodeElement {
                    name: t.ident.to_string(),
                    element_type: "trait".to_string(),
                    file_path: self.file_path.clone(),
                    line_number: 1,
                    parent: self.current_mod.last().cloned(),
                    visibility: self.format_visibility(&t.vis),
                    attributes: HashMap::new(),
                });
            }
            Item::Mod(m) => {
                if let Some(ident) = m.ident.as_ref() {
                    self.current_mod.push(ident.to_string());
                    syn::visit::visit_item(self, item);
                    self.current_mod.pop();
                    return;
                }
            }
            _ => {}
        }
        syn::visit::visit_item(self, item);
    }
}

impl AstVisitor {
    fn new(file_path: String) -> Self {
        Self {
            file_path,
            elements: Vec::new(),
            current_mod: Vec::new(),
        }
    }
    
    fn format_visibility(&self, vis: &syn::Visibility) -> String {
        match vis {
            syn::Visibility::Public(_) => "public".to_string(),
            syn::Visibility::Restricted(vis_restricted) => {
                if vis_restricted.path.is_ident("crate") {
                    "crate".to_string()
                } else if vis_restricted.path.is_ident("super") {
                    "super".to_string()
                } else {
                    "restricted".to_string()
                }
            }
            syn::Visibility::Inherited => "private".to_string(),
        }
    }
    
    fn extract_attributes(&self, attrs: &[syn::Attribute]) -> HashMap<String, String> {
        let mut attributes = HashMap::new();
        for attr in attrs {
            if let Ok(meta) = attr.meta.clone() {
                match meta {
                    syn::Meta::Path(path) => {
                        if let Some(ident) = path.get_ident() {
                            attributes.insert(ident.to_string(), "true".to_string());
                        }
                    }
                    syn::Meta::NameValue(name_value) => {
                        if let Some(ident) = name_value.path.get_ident() {
                            if let syn::Expr::Lit(expr_lit) = &name_value.value {
                                if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                    attributes.insert(ident.to_string(), lit_str.value());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        attributes
    }
}

impl GraphStore {
    pub async fn new(config: MemgraphConfig) -> Result<Self, GraphStoreError> {
        let graph = Graph::new(
            &config.url,
            config.username.as_deref().unwrap_or(""),
            config.password.as_deref().unwrap_or("")
        ).await?;
        
        // Test connection
        let mut result = graph.execute(query("RETURN 1 as test")).await?;
        if result.next().await?.is_some() {
            info!("Successfully connected to Memgraph/Neo4j");
        } else {
            return Err(GraphStoreError::Connection("Failed to verify connection".to_string()));
        }
        
        Ok(Self { graph, config })
    }
    
    pub async fn parse_repository(&self, repo_url: &str, branch: Option<&str>) -> Result<ParseRepositoryResult, GraphStoreError> {
        info!("Parsing repository: {} (branch: {:?})", repo_url, branch);
        
        // Clone repository to temporary directory
        let temp_dir = tempfile::tempdir()?;
        let repo_path = temp_dir.path();
        
        let mut clone_cmd = Command::new("git");
        clone_cmd.arg("clone");
        if let Some(b) = branch {
            clone_cmd.args(&["--branch", b]);
        }
        clone_cmd.arg(repo_url).arg(repo_path);
        
        let output = clone_cmd.output()?;
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(GraphStoreError::RepositoryClone(format!("Failed to clone repository: {}", error)));
        }
        
        // Clear existing repository data
        self.clear_repository_data(repo_url).await?;
        
        // Create repository node
        let repo_name = self.extract_repo_name(repo_url);
        self.graph.execute(query(
            "CREATE (r:Repository {name: $name, url: $url, branch: $branch})"
        ).param("name", repo_name.clone())
         .param("url", repo_url)
         .param("branch", branch.unwrap_or("main"))).await?;
        
        // Parse all source files
        let mut nodes_created = 1; // Repository node
        let mut relationships_created = 0;
        let mut files_processed = 0;
        
        for entry in WalkDir::new(repo_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            
            // Process supported source files
            match extension {
                "rs" | "py" | "js" | "ts" | "java" | "cpp" | "c" | "h" | "hpp" => {
                    match self.parse_source_file(path, &repo_name).await {
                        Ok((nodes, rels)) => {
                            nodes_created += nodes;
                            relationships_created += rels;
                            files_processed += 1;
                            debug!("Parsed file: {:?} ({} nodes, {} relationships)", path, nodes, rels);
                        }
                        Err(e) => {
                            warn!("Failed to parse file {:?}: {}", path, e);
                        }
                    }
                }
                _ => continue,
            }
        }
        
        info!(
            "Repository parsing completed: {} nodes, {} relationships, {} files",
            nodes_created, relationships_created, files_processed
        );
        
        Ok(ParseRepositoryResult {
            nodes_created,
            relationships_created,
            files_processed,
        })
    }
    
    async fn parse_source_file(&self, file_path: &Path, repo_name: &str) -> Result<(usize, usize), GraphStoreError> {
        let content = fs::read_to_string(file_path).await?;
        let relative_path = file_path.to_string_lossy();
        
        let mut nodes_created = 0;
        let mut relationships_created = 0;
        
        // Create file node
        self.graph.execute(query(
            "MERGE (f:File {path: $path, repository: $repo})"
        ).param("path", relative_path.as_ref())
         .param("repo", repo_name)).await?;
        
        nodes_created += 1;
        
        // Create relationship between repository and file
        self.graph.execute(query(
            "MATCH (r:Repository {name: $repo}), (f:File {path: $path, repository: $repo})
             MERGE (r)-[:CONTAINS]->(f)"
        ).param("repo", repo_name)
         .param("path", relative_path.as_ref())).await?;
        
        relationships_created += 1;
        
        // Parse based on file extension
        if relative_path.ends_with(".rs") {
            let (rust_nodes, rust_rels) = self.parse_rust_file(&content, &relative_path, repo_name).await?;
            nodes_created += rust_nodes;
            relationships_created += rust_rels;
        } else if relative_path.ends_with(".py") {
            let (py_nodes, py_rels) = self.parse_python_file(&content, &relative_path, repo_name).await?;
            nodes_created += py_nodes;
            relationships_created += py_rels;
        }
        // Add more language parsers as needed
        
        Ok((nodes_created, relationships_created))
    }
    
    async fn parse_rust_file(&self, content: &str, file_path: &str, repo_name: &str) -> Result<(usize, usize), GraphStoreError> {
        let mut nodes_created = 0;
        let mut relationships_created = 0;
        
        // Parse Rust AST
        match syn::parse_file(content) {
            Ok(file) => {
                let mut visitor = AstVisitor::new(file_path.to_string());
                visitor.visit_file(&file);
                
                for element in visitor.elements {
                    // Create nodes for code elements
                    self.graph.execute(query(
                        "CREATE (e:CodeElement {
                            name: $name,
                            type: $type,
                            file: $file,
                            line: $line,
                            visibility: $vis,
                            repository: $repo
                        })"
                    ).param("name", element.name.clone())
                     .param("type", element.element_type.clone())
                     .param("file", file_path)
                     .param("line", element.line_number as i64)
                     .param("vis", element.visibility.clone())
                     .param("repo", repo_name)).await?;
                    
                    nodes_created += 1;
                    
                    // Create relationship with file
                    self.graph.execute(query(
                        "MATCH (f:File {path: $path, repository: $repo}), 
                               (e:CodeElement {name: $name, file: $path, repository: $repo})
                         MERGE (f)-[:DEFINES]->(e)"
                    ).param("path", file_path)
                     .param("repo", repo_name)
                     .param("name", element.name.clone())).await?;
                    
                    relationships_created += 1;
                    
                    // Create parent relationships
                    if let Some(parent) = &element.parent {
                        self.graph.execute(query(
                            "MATCH (p:CodeElement {name: $parent, repository: $repo}),
                                   (c:CodeElement {name: $child, repository: $repo})
                             MERGE (p)-[:CONTAINS]->(c)"
                        ).param("parent", parent)
                         .param("child", element.name)
                         .param("repo", repo_name)).await?;
                        
                        relationships_created += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Failed to parse Rust file {}: {}", file_path, e);
            }
        }
        
        Ok((nodes_created, relationships_created))
    }
    
    async fn parse_python_file(&self, content: &str, file_path: &str, repo_name: &str) -> Result<(usize, usize), GraphStoreError> {
        let mut nodes_created = 0;
        let mut relationships_created = 0;
        
        // Simple regex-based Python parsing (could be improved with proper AST parsing)
        let function_re = Regex::new(r"^(\s*)def\s+(\w+)\s*\(")?;
        let class_re = Regex::new(r"^(\s*)class\s+(\w+)")?;
        
        for (line_num, line) in content.lines().enumerate() {
            if let Some(captures) = function_re.captures(line) {
                let func_name = &captures[2];
                
                self.graph.execute(query(
                    "CREATE (f:Function {
                        name: $name,
                        file: $file,
                        line: $line,
                        repository: $repo
                    })"
                ).param("name", func_name)
                 .param("file", file_path)
                 .param("line", line_num as i64 + 1)
                 .param("repo", repo_name)).await?;
                
                nodes_created += 1;
                
                // Create relationship with file
                self.graph.execute(query(
                    "MATCH (file:File {path: $path, repository: $repo}),
                           (func:Function {name: $name, file: $path, repository: $repo})
                     MERGE (file)-[:DEFINES]->(func)"
                ).param("path", file_path)
                 .param("repo", repo_name)
                 .param("name", func_name)).await?;
                
                relationships_created += 1;
            }
            
            if let Some(captures) = class_re.captures(line) {
                let class_name = &captures[2];
                
                self.graph.execute(query(
                    "CREATE (c:Class {
                        name: $name,
                        file: $file,
                        line: $line,
                        repository: $repo
                    })"
                ).param("name", class_name)
                 .param("file", file_path)
                 .param("line", line_num as i64 + 1)
                 .param("repo", repo_name)).await?;
                
                nodes_created += 1;
                
                self.graph.execute(query(
                    "MATCH (file:File {path: $path, repository: $repo}),
                           (class:Class {name: $name, file: $path, repository: $repo})
                     MERGE (file)-[:DEFINES]->(class)"
                ).param("path", file_path)
                 .param("repo", repo_name)
                 .param("name", class_name)).await?;
                
                relationships_created += 1;
            }
        }
        
        Ok((nodes_created, relationships_created))
    }
    
    pub async fn validate_script(&self, script_path: &str, context: Option<&str>) -> Result<ValidationResult, GraphStoreError> {
        info!("Validating script: {}", script_path);
        
        let content = fs::read_to_string(script_path).await?;
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();
        let mut is_valid = true;
        
        // Basic validation against knowledge graph
        if script_path.ends_with(".rs") {
            match syn::parse_file(&content) {
                Ok(file) => {
                    let mut visitor = AstVisitor::new(script_path.to_string());
                    visitor.visit_file(&file);
                    
                    for element in visitor.elements {
                        // Check if functions/structs exist in knowledge graph
                        let mut result = self.graph.execute(query(
                            "MATCH (e:CodeElement {name: $name, type: $type})
                             RETURN e.name as name, e.file as file"
                        ).param("name", element.name.clone())
                         .param("type", element.element_type.clone())).await?;
                        
                        if result.next().await?.is_none() {
                            // Element not found in knowledge graph
                            if element.element_type == "function" && !element.name.starts_with("test_") {
                                issues.push(ValidationIssue {
                                    line: element.line_number,
                                    column: 0,
                                    severity: "warning".to_string(),
                                    message: format!("Function '{}' not found in knowledge graph", element.name),
                                    suggestion: Some("This might be a hallucination. Verify function exists.".to_string()),
                                });
                                is_valid = false;
                            }
                        } else {
                            debug!("Validated element: {} ({})", element.name, element.element_type);
                        }
                    }
                }
                Err(e) => {
                    issues.push(ValidationIssue {
                        line: 1,
                        column: 0,
                        severity: "error".to_string(),
                        message: format!("Failed to parse Rust code: {}", e),
                        suggestion: Some("Check syntax errors in the code.".to_string()),
                    });
                    is_valid = false;
                }
            }
        }
        
        // Additional context-based validation
        if let Some(ctx) = context {
            if ctx.contains("repository") {
                suggestions.push("Consider validating against specific repository context.".to_string());
            }
        }
        
        let confidence = if is_valid { 
            1.0 - (issues.len() as f32 * 0.1).min(0.8)
        } else { 
            0.3 
        };
        
        Ok(ValidationResult {
            is_valid,
            confidence,
            issues,
            suggestions,
        })
    }
    
    pub async fn execute_query(&self, cypher_query: &str, limit: usize) -> Result<Vec<serde_json::Value>, GraphStoreError> {
        debug!("Executing Cypher query: {}", cypher_query);
        
        // Apply limit to query if not already present
        let query_with_limit = if cypher_query.to_uppercase().contains("LIMIT") {
            cypher_query.to_string()
        } else {
            format!("{} LIMIT {}", cypher_query, limit)
        };
        
        let mut result = self.graph.execute(query(&query_with_limit)).await?;
        let mut results = Vec::new();
        
        while let Some(row) = result.next().await? {
            let mut json_row = serde_json::Map::new();
            
            // Extract all columns from the row
            for key in row.keys() {
                if let Ok(value) = row.get::<neo4rs::BoltType>(key) {
                    match value {
                        neo4rs::BoltType::String(s) => {
                            json_row.insert(key.to_string(), serde_json::Value::String(s.value));
                        }
                        neo4rs::BoltType::Integer(i) => {
                            json_row.insert(key.to_string(), serde_json::Value::Number(
                                serde_json::Number::from(i.value)
                            ));
                        }
                        neo4rs::BoltType::Boolean(b) => {
                            json_row.insert(key.to_string(), serde_json::Value::Bool(b.value));
                        }
                        neo4rs::BoltType::Node(node) => {
                            let mut node_obj = serde_json::Map::new();
                            node_obj.insert("id".to_string(), serde_json::Value::Number(
                                serde_json::Number::from(node.id())
                            ));
                            node_obj.insert("labels".to_string(), serde_json::Value::Array(
                                node.labels().iter().map(|l| serde_json::Value::String(l.clone())).collect()
                            ));
                            
                            let mut props = serde_json::Map::new();
                            for (prop_key, prop_value) in node.properties() {
                                props.insert(prop_key.clone(), self.convert_bolt_value(prop_value));
                            }
                            node_obj.insert("properties".to_string(), serde_json::Value::Object(props));
                            
                            json_row.insert(key.to_string(), serde_json::Value::Object(node_obj));
                        }
                        _ => {
                            json_row.insert(key.to_string(), serde_json::Value::String(format!("{:?}", value)));
                        }
                    }
                }
            }
            
            results.push(serde_json::Value::Object(json_row));
        }
        
        info!("Query executed successfully, returned {} results", results.len());
        Ok(results)
    }
    
    fn convert_bolt_value(&self, value: &neo4rs::BoltType) -> serde_json::Value {
        match value {
            neo4rs::BoltType::String(s) => serde_json::Value::String(s.value.clone()),
            neo4rs::BoltType::Integer(i) => serde_json::Value::Number(serde_json::Number::from(i.value)),
            neo4rs::BoltType::Boolean(b) => serde_json::Value::Bool(b.value),
            _ => serde_json::Value::String(format!("{:?}", value)),
        }
    }
    
    async fn clear_repository_data(&self, repo_url: &str) -> Result<(), GraphStoreError> {
        self.graph.execute(query(
            "MATCH (r:Repository {url: $url})
             DETACH DELETE r"
        ).param("url", repo_url)).await?;
        
        Ok(())
    }
    
    fn extract_repo_name(&self, repo_url: &str) -> String {
        repo_url
            .split('/')
            .last()
            .unwrap_or("unknown")
            .trim_end_matches(".git")
            .to_string()
    }
}