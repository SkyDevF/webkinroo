# กินรู้ AI Model Specifications

## แนวคิดของแอป
- ใช้ AI วิเคราะห์ภาพอาหาร
- คำนวณแคลอรี่และสารอาหารอัตโนมัติ
- ช่วยให้ผู้ใช้ตัดสินใจเรื่องโภชนาการได้ง่ายขึ้น
- รองรับการติดตามสุขภาพ เช่น BMI

## ฟีเจอร์หลักของแอป
1. **กำหนดแคลอรี่ต่อวัน** - ตามค่า BMI ของผู้ใช้
2. **แสกนภาพอาหาร** - ระบบ AI วิเคราะห์เมนูและคำนวณแคลอรี่ทันที
3. **บันทึกมื้ออาหาร** - เก็บข้อมูลสารอาหารต่อวัน
4. **แนะนำเมนูอาหาร** - ตามเป้าหมายสุขภาพของผู้ใช้

## ผลการเทรนโมเดล AI

### Train Set Performance
- **Accuracy = 0.98 (98%)**
  - ความแม่นยำของโมเดลบน Train Set (ข้อมูลที่ใช้สอน)
  - หมายถึง โมเดลทำนายถูก 98 จาก 100 ตัวอย่าง

- **Loss = 0.12**
  - ค่า Loss บน Train Set คือค่าความผิดพลาดของโมเดล
  - ยิ่งน้อยยิ่งดี

### Validation Set Performance
- **Validation Accuracy = 0.73 (73%)**
  - ความแม่นยำของโมเดลบน Validation Set (ข้อมูลใหม่ที่ไม่ได้ใช้สอน)
  - โมเดลทำนายถูง 73 จาก 100 ตัวอย่าง

### Model Analysis
การที่ Train Accuracy (98%) สูงกว่า Validation Accuracy (73%) อย่างมีนัยสำคัญ แสดงให้เห็นว่าโมเดลมีปัญหา Overfitting ซึ่งเป็นเรื่องปกติในการพัฒนาโมเดล AI และสามารถปรับปรุงได้ด้วยเทคนิคต่างๆ เช่น:
- Data Augmentation
- Regularization
- Dropout
- Early Stopping

## AI Architecture Overview

### Multi-Stage AI Pipeline
1. **Image Preprocessing** → 2. **Food Detection** → 3. **Classification** → 4. **Portion Estimation** → 5. **Nutrition Calculation**

### Model Stack
- **Primary Classification**: Custom EfficientNet-B4 with Thai food specialization
- **Object Detection**: YOLOv8 for multi-food detection
- **Portion Estimation**: Depth estimation + reference object detection
- **Nutrition Calculation**: Ensemble model with nutritional database lookup

## Detailed Model Specifications

### 1. Food Detection Model (YOLOv8-Custom)

#### Architecture
```python
class FoodDetectionModel:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Base model
        self.input_size = (640, 640)
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        
    def detect_foods(self, image):
        """
        Detect and localize food items in image
        Returns: List of bounding boxes with confidence scores
        """
        results = self.model(image, conf=self.confidence_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
```

#### Training Configuration
- **Dataset**: 50,000+ Thai food images with bounding box annotations
- **Classes**: 200+ Thai food categories
- **Augmentation**: Rotation, scaling, color jittering, mixup
- **Training Time**: 100 epochs on 4x A100 GPUs
- **Validation Split**: 80/10/10 (train/val/test)

### 2. Food Classification Model (EfficientNet-B4 Custom)

#### Architecture
```python
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class ThaiFoodClassifier(nn.Module):
    def __init__(self, num_classes=500, dropout_rate=0.3):
        super(ThaiFoodClassifier, self).__init__()
        
        # Base EfficientNet-B4
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1792, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        
        # Auxiliary outputs for nutrition prediction
        self.nutrition_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)  # calories, protein, carbs, fat
        )
        
    def forward(self, x):
        features = self.backbone.extract_features(x)
        pooled = self.backbone._avg_pooling(features)
        pooled = pooled.flatten(start_dim=1)
        
        # Classification output
        class_logits = self.classifier(pooled)
        
        # Nutrition prediction
        nutrition_pred = self.nutrition_head(pooled[..., :512])
        
        return {
            'classification': class_logits,
            'nutrition': nutrition_pred,
            'features': pooled
        }
```

#### Training Strategy
```python
class TrainingConfig:
    # Model parameters
    input_size = (380, 380)
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Training schedule
    epochs = 150
    warmup_epochs = 10
    scheduler = 'cosine_annealing'
    
    # Loss functions
    classification_loss = 'cross_entropy'
    nutrition_loss = 'mse'
    loss_weights = {'classification': 1.0, 'nutrition': 0.5}
    
    # Data augmentation
    augmentations = [
        'random_rotation',
        'random_crop',
        'color_jitter',
        'gaussian_blur',
        'cutmix',
        'mixup'
    ]
```

### 3. Portion Estimation Model

#### Depth-Based Estimation
```python
class PortionEstimator:
    def __init__(self):
        self.depth_model = self.load_depth_model()
        self.reference_detector = self.load_reference_detector()
        
    def estimate_portion(self, image, food_bbox, food_class):
        """
        Estimate portion size using depth information and reference objects
        """
        # Extract food region
        food_region = self.extract_region(image, food_bbox)
        
        # Estimate depth
        depth_map = self.depth_model(food_region)
        
        # Detect reference objects (plates, utensils, hands)
        references = self.reference_detector(image)
        
        # Calculate scale factor
        scale_factor = self.calculate_scale(depth_map, references)
        
        # Estimate volume/weight
        volume = self.estimate_volume(food_region, depth_map, scale_factor)
        weight = self.volume_to_weight(volume, food_class)
        
        return {
            'weight_grams': weight,
            'volume_ml': volume,
            'confidence': self.calculate_confidence(references, depth_map)
        }
    
    def volume_to_weight(self, volume_ml, food_class):
        """Convert volume to weight using food density database"""
        density_map = {
            'rice': 1.5,  # g/ml
            'soup': 1.0,
            'meat': 1.2,
            'vegetables': 0.8,
            # ... more mappings
        }
        
        category = self.get_food_category(food_class)
        density = density_map.get(category, 1.0)
        
        return volume_ml * density
```

### 4. Nutrition Calculation Engine

#### Multi-Source Nutrition Database
```python
class NutritionCalculator:
    def __init__(self):
        self.food_db = self.load_food_database()
        self.ml_predictor = self.load_nutrition_predictor()
        
    def calculate_nutrition(self, food_id, portion_grams):
        """
        Calculate nutrition using multiple sources and ML prediction
        """
        # Method 1: Database lookup
        db_nutrition = self.get_db_nutrition(food_id, portion_grams)
        
        # Method 2: ML prediction from image features
        ml_nutrition = self.ml_predictor.predict(food_id, portion_grams)
        
        # Method 3: Ingredient-based calculation
        ingredient_nutrition = self.calculate_from_ingredients(food_id, portion_grams)
        
        # Ensemble prediction with confidence weighting
        final_nutrition = self.ensemble_predict([
            (db_nutrition, 0.6),
            (ml_nutrition, 0.3),
            (ingredient_nutrition, 0.1)
        ])
        
        return final_nutrition
    
    def ensemble_predict(self, predictions):
        """Weighted ensemble of nutrition predictions"""
        total_weight = sum(weight for _, weight in predictions)
        
        result = {
            'calories': 0,
            'protein': 0,
            'carbohydrates': 0,
            'fat': 0,
            'fiber': 0,
            'sugar': 0,
            'sodium': 0
        }
        
        for nutrition, weight in predictions:
            if nutrition:
                for key in result:
                    result[key] += nutrition.get(key, 0) * weight / total_weight
        
        return result
```

### 5. Model Optimization for Mobile

#### TensorFlow Lite Conversion
```python
def convert_to_tflite(model_path, output_path):
    """Convert PyTorch model to TensorFlow Lite for mobile deployment"""
    
    # Convert to ONNX first
    dummy_input = torch.randn(1, 3, 380, 380)
    torch.onnx.export(model, dummy_input, "model.onnx")
    
    # Convert ONNX to TensorFlow
    import onnx_tf
    onnx_model = onnx.load("model.onnx")
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph("model.pb")
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model("model.pb")
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Quantization for smaller model size
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

#### Model Compression Techniques
```python
class ModelCompression:
    @staticmethod
    def knowledge_distillation(teacher_model, student_model, dataloader):
        """Compress model using knowledge distillation"""
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        criterion_ce = nn.CrossEntropyLoss()
        
        for batch in dataloader:
            images, labels = batch
            
            # Teacher predictions (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            # Student predictions
            student_outputs = student_model(images)
            
            # Distillation loss
            kd_loss = criterion_kd(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            )
            
            # Classification loss
            ce_loss = criterion_ce(student_outputs, labels)
            
            # Combined loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    @staticmethod
    def pruning(model, sparsity=0.3):
        """Prune model weights to reduce size"""
        import torch.nn.utils.prune as prune
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
```

## Training Data Pipeline

### Data Collection Strategy
```python
class DataCollectionPipeline:
    def __init__(self):
        self.sources = [
            'restaurant_menus',
            'food_delivery_apps',
            'social_media',
            'user_submissions',
            'professional_photography'
        ]
    
    def collect_thai_food_data(self):
        """Collect and curate Thai food dataset"""
        
        # 1. Web scraping from food websites
        scraped_data = self.scrape_food_websites()
        
        # 2. API integration with food databases
        api_data = self.fetch_from_apis()
        
        # 3. User-generated content
        user_data = self.collect_user_submissions()
        
        # 4. Professional food photography
        professional_data = self.license_professional_photos()
        
        # 5. Data validation and cleaning
        cleaned_data = self.validate_and_clean([
            scraped_data, api_data, user_data, professional_data
        ])
        
        return cleaned_data
    
    def augment_dataset(self, base_dataset):
        """Apply data augmentation to increase dataset size"""
        augmentations = [
            self.lighting_variations,
            self.angle_variations,
            self.background_variations,
            self.portion_variations,
            self.presentation_variations
        ]
        
        augmented_data = []
        for image, label in base_dataset:
            # Original image
            augmented_data.append((image, label))
            
            # Apply augmentations
            for aug_func in augmentations:
                aug_image = aug_func(image)
                augmented_data.append((aug_image, label))
        
        return augmented_data
```

### Annotation Pipeline
```python
class AnnotationPipeline:
    def __init__(self):
        self.annotation_tools = ['labelme', 'cvat', 'custom_tool']
        self.quality_threshold = 0.95
    
    def create_annotations(self, images):
        """Create high-quality annotations for training data"""
        
        annotations = []
        for image in images:
            # 1. Automatic pre-annotation using existing models
            pre_annotation = self.pre_annotate(image)
            
            # 2. Human annotation and correction
            human_annotation = self.human_annotate(image, pre_annotation)
            
            # 3. Quality validation
            if self.validate_quality(human_annotation) > self.quality_threshold:
                annotations.append(human_annotation)
            else:
                # Re-annotate if quality is insufficient
                annotations.append(self.re_annotate(image))
        
        return annotations
    
    def validate_quality(self, annotation):
        """Validate annotation quality using multiple metrics"""
        metrics = {
            'bbox_accuracy': self.check_bbox_accuracy(annotation),
            'label_consistency': self.check_label_consistency(annotation),
            'completeness': self.check_completeness(annotation)
        }
        
        return sum(metrics.values()) / len(metrics)
```

## Model Evaluation and Monitoring

### Evaluation Metrics
```python
class ModelEvaluator:
    def __init__(self):
        self.metrics = {
            'classification': ['accuracy', 'precision', 'recall', 'f1'],
            'detection': ['mAP', 'precision', 'recall'],
            'nutrition': ['mae', 'mape', 'r2_score']
        }
    
    def evaluate_classification(self, model, test_loader):
        """Evaluate food classification performance"""
        y_true, y_pred = [], []
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                predictions = torch.argmax(outputs['classification'], dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def evaluate_nutrition_accuracy(self, predictions, ground_truth):
        """Evaluate nutrition prediction accuracy"""
        metrics = {}
        
        for nutrient in ['calories', 'protein', 'carbs', 'fat']:
            pred_values = [p[nutrient] for p in predictions]
            true_values = [gt[nutrient] for gt in ground_truth]
            
            metrics[f'{nutrient}_mae'] = mean_absolute_error(true_values, pred_values)
            metrics[f'{nutrient}_mape'] = mean_absolute_percentage_error(true_values, pred_values)
            metrics[f'{nutrient}_r2'] = r2_score(true_values, pred_values)
        
        return metrics
```

### Continuous Learning Pipeline
```python
class ContinuousLearning:
    def __init__(self):
        self.feedback_threshold = 100  # Minimum feedback samples for retraining
        self.performance_threshold = 0.85  # Minimum acceptable performance
    
    def collect_user_feedback(self, prediction_id, user_correction):
        """Collect user feedback for model improvement"""
        feedback = {
            'prediction_id': prediction_id,
            'original_prediction': self.get_prediction(prediction_id),
            'user_correction': user_correction,
            'timestamp': datetime.now(),
            'confidence_delta': self.calculate_confidence_delta(prediction_id, user_correction)
        }
        
        self.store_feedback(feedback)
        
        # Trigger retraining if enough feedback collected
        if self.get_feedback_count() >= self.feedback_threshold:
            self.trigger_retraining()
    
    def adaptive_model_update(self):
        """Update model based on recent performance and feedback"""
        recent_performance = self.evaluate_recent_performance()
        
        if recent_performance < self.performance_threshold:
            # Collect additional training data
            new_data = self.collect_targeted_data()
            
            # Fine-tune model
            self.fine_tune_model(new_data)
            
            # Validate improvements
            if self.validate_improvements():
                self.deploy_updated_model()
            else:
                self.rollback_model()
```

This comprehensive AI model specification provides the foundation for building a robust, accurate, and scalable food recognition system specifically tailored for Thai cuisine and nutrition tracking.