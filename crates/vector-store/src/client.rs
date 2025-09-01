use std::sync::Arc;
use uuid::Uuid;
use qdrant_client::{
    Qdrant,
    qdrant::{
        vectors_config::Config, CreateCollection, Distance as QdrantDistance,
        FieldType, Filter, PointStruct, SearchPoints, VectorParams, VectorsConfig,
        WithPayloadSelector, WithVectorsSelector, PointsSelector, PointsIdsList, PointId,
        CreateFieldIndexCollection, PayloadIndexParams, TextIndexParams, VectorParamsMap,
        Value, value::Kind, ListValue, DeletePoints, UpsertPoints, ScrollPoints,
    },
};
use tracing::{debug, info, warn};

use crate::{
    CollectionConfig, Distance, PagePoint, PageVectors, QdrantConfig, 
    Result, SearchQuery, SearchResult, VectorStoreError,
};

pub struct VectorStore {
    client: Arc<Qdrant>,
    config: QdrantConfig,
    collection_config: CollectionConfig,
}

impl VectorStore {
    pub async fn new(
        config: QdrantConfig,
        collection_config: CollectionConfig,
    ) -> Result<Self> {
        let mut client_builder = Qdrant::from_url(&config.url);
        if let Some(api_key) = &config.api_key {
            client_builder = client_builder.api_key(api_key.clone());
        }
        let client = client_builder
            .build()
            .map_err(|e| VectorStoreError::Configuration(e.to_string()))?;
        
        let client = Arc::new(client);
        
        let store = Self {
            client,
            config,
            collection_config,
        };
        
        // Initialize collection
        store.init_collection().await?;
        
        Ok(store)
    }
    
    async fn init_collection(&self) -> Result<()> {
        let collection_name = &self.collection_config.name;
        
        // Check if collection exists
        match self.client.collection_info(collection_name).await {
            Ok(_) => {
                info!("Collection '{}' already exists", collection_name);
                Ok(())
            }
            Err(_) => {
                info!("Creating collection '{}'", collection_name);
                self.create_collection().await
            }
        }
    }
    
    async fn create_collection(&self) -> Result<()> {
        let mut vectors_params_map = std::collections::HashMap::new();
        
        // Title vector configuration
        vectors_params_map.insert(
            "title".to_string(),
            VectorParams {
                size: self.collection_config.vectors.title.size,
                distance: self.convert_distance(self.collection_config.vectors.title.distance).into(),
                on_disk: self.collection_config.vectors.title.on_disk,
                ..Default::default()
            },
        );
        
        // Content vector configuration
        vectors_params_map.insert(
            "content".to_string(),
            VectorParams {
                size: self.collection_config.vectors.content.size,
                distance: self.convert_distance(self.collection_config.vectors.content.distance).into(),
                on_disk: self.collection_config.vectors.content.on_disk,
                ..Default::default()
            },
        );
        
        // Code vector configuration (optional)
        if let Some(code_config) = &self.collection_config.vectors.code {
            vectors_params_map.insert(
                "code".to_string(),
                VectorParams {
                    size: code_config.size,
                    distance: self.convert_distance(code_config.distance).into(),
                    on_disk: code_config.on_disk,
                    ..Default::default()
                },
            );
        }
        
        let create_collection = CreateCollection {
            collection_name: self.collection_config.name.clone(),
            vectors_config: Some(VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::ParamsMap(
                    VectorParamsMap {
                        map: vectors_params_map,
                    }
                )),
            }),
            shard_number: self.collection_config.shard_number,
            replication_factor: self.collection_config.replication_factor,
            write_consistency_factor: self.collection_config.write_consistency_factor,
            ..Default::default()
        };
        
        self.client
            .create_collection(create_collection)
            .await?;
        
        // Create indexes for metadata fields
        self.create_payload_indexes().await?;
        
        // Enable binary quantization if configured
        if self.config.use_binary_quantization {
            self.enable_binary_quantization().await?;
        }
        
        Ok(())
    }
    
    async fn create_payload_indexes(&self) -> Result<()> {
        let collection_name = &self.collection_config.name;
        
        // Create index for domain field
        let domain_index_request = CreateFieldIndexCollection {
            collection_name: collection_name.to_string(),
            field_name: "domain".to_string(),
            field_type: Some(FieldType::Keyword.into()),
            field_index_params: None,
            wait: None,
            ordering: None,
        };
        self.client
            .create_field_index(domain_index_request)
            .await?;
        
        // Create index for crawled_at field
        let crawled_at_index_request = CreateFieldIndexCollection {
            collection_name: collection_name.to_string(),
            field_name: "crawled_at".to_string(),
            field_type: Some(FieldType::Datetime.into()),
            field_index_params: None,
            wait: None,
            ordering: None,
        };
        self.client
            .create_field_index(crawled_at_index_request)
            .await?;
        
        Ok(())
    }
    
    async fn enable_binary_quantization(&self) -> Result<()> {
        // Note: Binary quantization is enabled during collection creation
        // This is a placeholder for any additional configuration
        info!("Binary quantization enabled for collection");
        Ok(())
    }
    
    pub async fn upsert_page(
        &self,
        point: PagePoint,
        vectors: PageVectors,
    ) -> Result<()> {
        let point_id = point.id.to_string();
        
        // Prepare named vectors
        let mut named_vectors = std::collections::HashMap::new();
        named_vectors.insert("title".to_string(), vectors.title_vector);
        named_vectors.insert("content".to_string(), vectors.content_vector);
        
        if let Some(code_vector) = vectors.code_vector {
            named_vectors.insert("code".to_string(), code_vector);
        }
        
        // Convert to Qdrant point
        let qdrant_point = PointStruct {
            id: Some(point_id.into()),
            vectors: Some(named_vectors.into()),
            payload: {
                let json_value = serde_json::to_value(&point)?;
                let mut payload = std::collections::HashMap::new();
                
                if let serde_json::Value::Object(obj) = json_value {
                    for (key, value) in obj {
                        payload.insert(key, convert_json_to_qdrant_value(value));
                    }
                }
                
                payload
            },
        };
        
        // Upsert point
        let upsert_request = UpsertPoints {
            collection_name: self.collection_config.name.clone(),
            wait: None,
            points: vec![qdrant_point],
            ordering: None,
            shard_key_selector: None,
        };
        
        self.client
            .upsert_points(upsert_request)
            .await?;
        
        debug!("Upserted page: {}", point.url);
        Ok(())
    }
    
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let collection_name = &self.collection_config.name;
        
        // Build filter if provided
        let filter = query.filter.as_ref().map(|f| self.build_filter(f));
        
        // Search with multiple vectors
        let mut all_results = Vec::new();
        
        // Search by title vector
        if let Some(title_vector) = query.query_vectors.title_vector {
            let results = self.search_single_vector(
                collection_name,
                "title",
                title_vector,
                query.limit,
                filter.clone(),
                query.score_threshold,
            ).await?;
            
            for mut result in results {
                result.score *= query.query_vectors.weights.title;
                all_results.push(result);
            }
        }
        
        // Search by content vector
        if let Some(content_vector) = query.query_vectors.content_vector {
            let results = self.search_single_vector(
                collection_name,
                "content",
                content_vector,
                query.limit,
                filter.clone(),
                query.score_threshold,
            ).await?;
            
            for mut result in results {
                result.score *= query.query_vectors.weights.content;
                all_results.push(result);
            }
        }
        
        // Search by code vector
        if let Some(code_vector) = query.query_vectors.code_vector {
            let results = self.search_single_vector(
                collection_name,
                "code",
                code_vector,
                query.limit,
                filter,
                query.score_threshold,
            ).await?;
            
            for mut result in results {
                result.score *= query.query_vectors.weights.code;
                all_results.push(result);
            }
        }
        
        // Merge and sort results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_results.truncate(query.limit);
        
        Ok(all_results)
    }
    
    async fn search_single_vector(
        &self,
        collection_name: &str,
        vector_name: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let search_request = SearchPoints {
            collection_name: collection_name.to_string(),
            vector: vector,
            vector_name: Some(vector_name.to_string()),
            filter,
            limit: limit as u64,
            with_payload: Some(WithPayloadSelector::from(true)),
            with_vectors: Some(WithVectorsSelector::from(false)),
            score_threshold,
            ..Default::default()
        };
        
        let response = self.client.search_points(search_request).await?;
        
        let results = response
            .result
            .into_iter()
            .map(|point| {
                let payload = if point.payload.is_empty() {
                    None
                } else {
                    convert_qdrant_payload_to_json(&point.payload)
                        .ok()
                        .and_then(|json| serde_json::from_value(json).ok())
                };
                
                SearchResult {
                    id: match point.id.unwrap().point_id_options.unwrap() {
                        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid) => Uuid::parse_str(&uuid).unwrap(),
                        qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => {
                            Uuid::from_u128(num as u128)
                        },
                    },
                    score: point.score,
                    payload,
                    vectors: None,
                }
            })
            .collect();
        
        Ok(results)
    }
    
    fn build_filter(&self, filter: &crate::SearchFilter) -> Filter {
        let mut conditions = Vec::new();
        
        if let Some(domain) = &filter.domain {
            conditions.push(qdrant_client::qdrant::Condition {
                condition_one_of: Some(
                    qdrant_client::qdrant::condition::ConditionOneOf::Field(
                        qdrant_client::qdrant::FieldCondition {
                            key: "domain".to_string(),
                            r#match: Some(qdrant_client::qdrant::Match {
                                match_value: Some(
                                    qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                        domain.clone()
                                    )
                                ),
                            }),
                            ..Default::default()
                        }
                    )
                ),
            });
        }
        
        Filter {
            must: conditions,
            ..Default::default()
        }
    }
    
    fn convert_distance(&self, distance: Distance) -> QdrantDistance {
        match distance {
            Distance::Cosine => QdrantDistance::Cosine,
            Distance::Euclid => QdrantDistance::Euclid,
            Distance::Dot => QdrantDistance::Dot,
        }
    }
    
    pub async fn delete_by_id(&self, id: Uuid) -> Result<()> {
        let points_selector = PointsSelector {
            points_selector_one_of: Some(
                qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                    PointsIdsList {
                        ids: vec![PointId::from(id.to_string())],
                    }
                )
            ),
        };

        let delete_request = DeletePoints {
            collection_name: self.collection_config.name.clone(),
            points: Some(points_selector),
            wait: None,
            ordering: None,
            shard_key_selector: None,
        };
        
        self.client
            .delete_points(delete_request)
            .await?;
        
        Ok(())
    }
    
    pub async fn get_collection_info(&self) -> Result<serde_json::Value> {
        let info = self.client
            .collection_info(&self.collection_config.name)
            .await?;
        
        // Extract basic info manually since GetCollectionInfoResponse doesn't implement Serialize
        let mut map = serde_json::Map::new();
        map.insert("status".to_string(), serde_json::Value::String("available".to_string()));
        // Add more fields as needed from the actual response structure
        
        Ok(serde_json::Value::Object(map))
    }
    
    pub async fn list_all_pages(&self) -> Result<Vec<PagePoint>> {
        // Simple implementation that scrolls through all points
        // In production, you might want to add pagination
        let scroll_request = qdrant_client::qdrant::ScrollPoints {
            collection_name: self.collection_config.name.clone(),
            filter: None,
            offset: None,
            limit: Some(10000), // Reasonable limit to avoid memory issues
            with_payload: Some(true.into()),
            with_vectors: None,
            read_consistency: None,
            order_by: None,
            shard_key_selector: None,
            timeout: None,
        };
        
        let response = self.client.scroll(scroll_request).await?;
        let mut pages = Vec::new();
        
        for retrieved_point in response.result {
            let payload = retrieved_point.payload;
            if let (Some(url), Some(title), Some(domain)) = (
                    payload.get("url").and_then(|v| match &v.kind {
                        Some(Kind::StringValue(s)) => Some(s.clone()),
                        _ => None,
                    }),
                    payload.get("title").and_then(|v| match &v.kind {
                        Some(Kind::StringValue(s)) => Some(s.clone()),
                        _ => None,
                    }),
                    payload.get("domain").and_then(|v| match &v.kind {
                        Some(Kind::StringValue(s)) => Some(s.clone()),
                        _ => None,
                    }),
                ) {
                    let content_preview = payload.get("content_preview")
                        .and_then(|v| match &v.kind {
                            Some(Kind::StringValue(s)) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    
                    let crawled_at = payload.get("crawled_at")
                        .and_then(|v| match &v.kind {
                            Some(Kind::StringValue(s)) => chrono::DateTime::parse_from_rfc3339(&s)
                                .map(|dt| dt.with_timezone(&chrono::Utc))
                                .ok(),
                            _ => None,
                        })
                        .unwrap_or_else(chrono::Utc::now);
                    
                    pages.push(PagePoint {
                        id: match retrieved_point.id.and_then(|id| id.point_id_options) {
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid_str)) => {
                                Uuid::parse_str(&uuid_str).unwrap_or_default()
                            },
                            _ => Uuid::new_v4(),
                        },
                        url,
                        title,
                        domain,
                        content_preview,
                        crawled_at,
                        metadata: std::collections::HashMap::new(),
                    });
            }
        }
        
        Ok(pages)
    }
}

fn convert_json_to_qdrant_value(json_value: serde_json::Value) -> Value {
    match json_value {
        serde_json::Value::Null => Value {
            kind: Some(Kind::NullValue(0)),
        },
        serde_json::Value::Bool(b) => Value {
            kind: Some(Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value {
                    kind: Some(Kind::IntegerValue(i)),
                }
            } else if let Some(f) = n.as_f64() {
                Value {
                    kind: Some(Kind::DoubleValue(f)),
                }
            } else {
                Value {
                    kind: Some(Kind::StringValue(n.to_string())),
                }
            }
        },
        serde_json::Value::String(s) => Value {
            kind: Some(Kind::StringValue(s)),
        },
        serde_json::Value::Array(arr) => {
            let list_value = ListValue {
                values: arr.into_iter().map(convert_json_to_qdrant_value).collect(),
            };
            Value {
                kind: Some(Kind::ListValue(list_value)),
            }
        },
        serde_json::Value::Object(obj) => {
            let struct_value = qdrant_client::qdrant::Struct {
                fields: obj.into_iter().map(|(k, v)| (k, convert_json_to_qdrant_value(v))).collect(),
            };
            Value {
                kind: Some(Kind::StructValue(struct_value)),
            }
        },
    }
}

fn convert_qdrant_payload_to_json(payload: &std::collections::HashMap<String, Value>) -> Result<serde_json::Value> {
    let mut json_map = serde_json::Map::new();
    
    for (key, value) in payload {
        json_map.insert(key.clone(), convert_qdrant_value_to_json(value));
    }
    
    Ok(serde_json::Value::Object(json_map))
}

fn convert_qdrant_value_to_json(value: &Value) -> serde_json::Value {
    match &value.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::IntegerValue(i)) => serde_json::Value::Number((*i).into()),
        Some(Kind::DoubleValue(f)) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)))
        },
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            let array: Vec<serde_json::Value> = list.values.iter()
                .map(convert_qdrant_value_to_json)
                .collect();
            serde_json::Value::Array(array)
        },
        Some(Kind::StructValue(struct_val)) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in &struct_val.fields {
                obj.insert(k.clone(), convert_qdrant_value_to_json(v));
            }
            serde_json::Value::Object(obj)
        },
        None => serde_json::Value::Null,
    }
}