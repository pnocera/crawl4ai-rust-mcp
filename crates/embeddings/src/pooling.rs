use candle_core::{Device, Tensor, D};
use tracing::debug;
use ndarray;

use crate::{EmbeddingsError, Result};

// Normalization strategies
pub fn normalize_embeddings(embeddings: &Tensor, p: f32) -> Result<Tensor> {
    if p == 2.0 {
        // L2 normalization (most common)
        l2_normalize(embeddings)
    } else if p == 1.0 {
        // L1 normalization
        l1_normalize(embeddings)
    } else {
        // Lp normalization
        lp_normalize(embeddings, p)
    }
}

pub fn l2_normalize(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    
    Ok((embeddings / norm)?)
}

pub fn l1_normalize(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings
        .abs()?
        .sum_keepdim(D::Minus1)?
        .clamp(1e-12, f64::INFINITY)?;
    
    Ok((embeddings / norm)?)
}

pub fn lp_normalize(embeddings: &Tensor, p: f32) -> Result<Tensor> {
    let norm = embeddings
        .abs()?
        .powf(p as f64)?
        .sum_keepdim(D::Minus1)?
        .powf(1.0 / p as f64)?
        .clamp(1e-12, f64::INFINITY)?;
    
    Ok((embeddings / norm)?)
}

// Whitening transformation for better isotropy
pub struct WhiteningTransform {
    mean: Tensor,
    components: Tensor,
    device: Device,
}

impl WhiteningTransform {
    pub fn fit(embeddings: &Tensor) -> Result<Self> {
        let device = embeddings.device().clone();
        let (batch_size, hidden_size) = embeddings.dims2()?;
        
        // Calculate mean
        let mean = embeddings.mean(0)?;
        
        // Center the embeddings
        let centered = embeddings.broadcast_sub(&mean)?;
        
        // Calculate covariance matrix
        let cov = centered.t()?.matmul(&centered)? / (batch_size as f64 - 1.0);
        
        // Convert to ndarray for proper linear algebra operations
        let cov = cov?;
        let cov_shape = cov.shape();
        let cov_data: Vec<f32> = cov.to_vec1()?;
        let cov_ndarray = ndarray::Array2::from_shape_vec((cov_shape.dims()[0], cov_shape.dims()[1]), cov_data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to convert to ndarray: {}", e)))?;
        
        // Perform eigendecomposition using ndarray-linalg would be ideal, but we'll use a simpler approach
        // For now, use SVD approximation which is more stable than eigendecomposition
        let (u, s, _vt) = Self::svd_decomposition(&cov_ndarray)?;
        
        // Select top components (PCA dimensionality reduction could be applied here)
        let components_data: Vec<f32> = u.iter().cloned().collect();
        let components = Tensor::from_vec(components_data, (hidden_size, hidden_size), &device)?;
        
        debug!("Fitted whitening transform on {} embeddings", batch_size);
        
        Ok(Self {
            mean,
            components,
            device,
        })
    }
    
    fn svd_decomposition(matrix: &ndarray::Array2<f32>) -> Result<(ndarray::Array2<f32>, ndarray::Array1<f32>, ndarray::Array2<f32>)> {
        // Simple SVD implementation using power iteration method
        // For production use, consider using ndarray-linalg or faer for proper SVD
        
        let (rows, cols) = matrix.dim();
        let min_dim = rows.min(cols);
        
        // Initialize U, S, V matrices
        let mut u = ndarray::Array2::<f32>::eye(rows);
        let mut s = ndarray::Array1::<f32>::zeros(min_dim);
        let vt = ndarray::Array2::<f32>::eye(cols);
        
        // For simplicity, compute only the diagonal of the SVD
        // This is not a full SVD but provides a reasonable approximation for whitening
        let aat = matrix.dot(&matrix.t());
        
        // Power iteration to find the largest eigenvalue/eigenvector
        for i in 0..min_dim {
            let mut v = ndarray::Array1::<f32>::ones(rows);
            
            // Power iteration
            for _ in 0..10 { // 10 iterations should be enough for convergence
                v = aat.dot(&v);
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    v /= norm;
                }
            }
            
            // Compute singular value
            let sv = aat.dot(&v);
            let sigma = sv.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>().sqrt();
            s[i] = sigma;
            
            // Store eigenvector in U
            for j in 0..rows {
                u[[j, i]] = v[j];
            }
        }
        
        Ok((u, s, vt))
    }
    
    pub fn transform(&self, embeddings: &Tensor) -> Result<Tensor> {
        // Center
        let centered = embeddings.broadcast_sub(&self.mean)?;
        
        // Apply whitening
        let whitened = centered.matmul(&self.components)?;
        
        Ok(whitened)
    }
}

// Quantization for memory efficiency
pub fn quantize_embeddings(embeddings: &Tensor, bits: u8) -> Result<QuantizedEmbeddings> {
    match bits {
        1 => binary_quantize(embeddings),
        8 => int8_quantize(embeddings),
        _ => Err(EmbeddingsError::InvalidInput(
            format!("Unsupported quantization bits: {}", bits)
        )),
    }
}

pub struct QuantizedEmbeddings {
    pub data: Vec<u8>,
    pub scale: Vec<f32>,
    pub zero_point: Vec<f32>,
    pub shape: Vec<usize>,
    pub bits: u8,
}

fn binary_quantize(embeddings: &Tensor) -> Result<QuantizedEmbeddings> {
    let shape = embeddings.shape().dims().to_vec();
    let flat = embeddings.flatten_all()?;
    let values = flat.to_vec1::<f32>()?;
    
    let mut quantized = vec![0u8; (values.len() + 7) / 8];
    
    for (i, &value) in values.iter().enumerate() {
        if value > 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            quantized[byte_idx] |= 1 << bit_idx;
        }
    }
    
    Ok(QuantizedEmbeddings {
        data: quantized,
        scale: vec![1.0],
        zero_point: vec![0.0],
        shape,
        bits: 1,
    })
}

fn int8_quantize(embeddings: &Tensor) -> Result<QuantizedEmbeddings> {
    let shape = embeddings.shape().dims().to_vec();
    let values = embeddings.flatten_all()?.to_vec1::<f32>()?;
    
    // Find min and max
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate scale and zero point
    let scale = (max_val - min_val) / 255.0;
    let zero_point = -min_val / scale;
    
    // Quantize
    let quantized: Vec<u8> = values
        .iter()
        .map(|&v| ((v - min_val) / scale).round() as u8)
        .collect();
    
    Ok(QuantizedEmbeddings {
        data: quantized,
        scale: vec![scale],
        zero_point: vec![zero_point],
        shape,
        bits: 8,
    })
}

impl QuantizedEmbeddings {
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        match self.bits {
            1 => self.dequantize_binary(device),
            8 => self.dequantize_int8(device),
            _ => Err(EmbeddingsError::InvalidInput(
                format!("Unsupported quantization bits: {}", self.bits)
            )),
        }
    }
    
    fn dequantize_binary(&self, device: &Device) -> Result<Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let mut values = Vec::with_capacity(total_elements);
        
        for i in 0..total_elements {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            
            let value = if byte_idx < self.data.len() && 
                (self.data[byte_idx] & (1 << bit_idx)) != 0 {
                1.0
            } else {
                -1.0
            };
            
            values.push(value);
        }
        
        Tensor::from_vec(values, self.shape.clone(), device)
            .map_err(EmbeddingsError::from)
    }
    
    fn dequantize_int8(&self, device: &Device) -> Result<Tensor> {
        let scale = self.scale[0];
        let zero_point = self.zero_point[0];
        
        let values: Vec<f32> = self.data
            .iter()
            .map(|&q| (q as f32 - zero_point) * scale)
            .collect();
        
        Tensor::from_vec(values, self.shape.clone(), device)
            .map_err(EmbeddingsError::from)
    }
    
    pub fn memory_usage(&self) -> usize {
        self.data.len() + 
        self.scale.len() * std::mem::size_of::<f32>() +
        self.zero_point.len() * std::mem::size_of::<f32>() +
        self.shape.len() * std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_normalization() {
        let device = Device::Cpu;
        let embeddings = Tensor::new(&[[3.0f32, 4.0], [5.0, 12.0]], &device).unwrap();
        
        let normalized = l2_normalize(&embeddings).unwrap();
        let norms = normalized.sqr().unwrap().sum(D::Minus1).unwrap().sqrt().unwrap();
        
        // All norms should be ~1.0
        let norm_values = norms.to_vec1::<f32>().unwrap();
        for norm in norm_values {
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_binary_quantization() {
        let device = Device::Cpu;
        let embeddings = Tensor::new(&[[0.5f32, -0.3], [-0.8, 0.2]], &device).unwrap();
        
        let quantized = binary_quantize(&embeddings).unwrap();
        let dequantized = quantized.dequantize(&device).unwrap();
        
        // Check signs are preserved
        let original = embeddings.to_vec2::<f32>().unwrap();
        let deq = dequantized.to_vec2::<f32>().unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(original[i][j].signum(), deq[i][j]);
            }
        }
    }
}