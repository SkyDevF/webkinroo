# กินรู้ Database Schema Design

## Database Architecture Overview

### Primary Databases
- **PostgreSQL**: Relational data (users, food database, logs)
- **MongoDB**: Document storage (AI analysis results, recommendations)
- **Redis**: Caching and session management
- **InfluxDB**: Time-series data (weight tracking, analytics)

## PostgreSQL Schema

### Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    
    -- Indexes
    CONSTRAINT users_email_key UNIQUE (email)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);
```

### User Profiles Table
```sql
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    gender VARCHAR(10) CHECK (gender IN ('male', 'female', 'other')),
    height_cm DECIMAL(5,2),
    current_weight_kg DECIMAL(5,2),
    
    -- BMI และการคำนวณแคลอรี่
    current_bmi DECIMAL(4,2) GENERATED ALWAYS AS (
        CASE 
            WHEN height_cm > 0 AND current_weight_kg > 0 
            THEN current_weight_kg / POWER(height_cm / 100, 2)
            ELSE NULL 
        END
    ) STORED,
    bmi_category VARCHAR(20) GENERATED ALWAYS AS (
        CASE 
            WHEN current_bmi < 18.5 THEN 'underweight'
            WHEN current_bmi >= 18.5 AND current_bmi < 25 THEN 'normal'
            WHEN current_bmi >= 25 AND current_bmi < 30 THEN 'overweight'
            WHEN current_bmi >= 30 THEN 'obese'
            ELSE 'unknown'
        END
    ) STORED,
    
    -- BMR และ TDEE calculation
    bmr_calories DECIMAL(7,2), -- Basal Metabolic Rate
    tdee_calories DECIMAL(7,2), -- Total Daily Energy Expenditure
    recommended_daily_calories INTEGER, -- แคลอรี่ที่แนะนำต่อวัน
    
    activity_level VARCHAR(20) CHECK (activity_level IN ('sedentary', 'light', 'moderate', 'active', 'very_active')),
    timezone VARCHAR(50) DEFAULT 'Asia/Bangkok',
    language VARCHAR(10) DEFAULT 'th',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT user_profiles_user_id_key UNIQUE (user_id)
);

CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_profiles_bmi_category ON user_profiles(bmi_category);
```

### Health Conditions Table
```sql
CREATE TABLE health_conditions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    condition_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('mild', 'moderate', 'severe')),
    diagnosed_date DATE,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_health_conditions_user_id ON health_conditions(user_id);
CREATE INDEX idx_health_conditions_type ON health_conditions(condition_type);
```

### Allergies and Restrictions Table
```sql
CREATE TABLE dietary_restrictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    restriction_type VARCHAR(20) CHECK (restriction_type IN ('allergy', 'intolerance', 'preference', 'religious', 'medical')),
    item VARCHAR(100) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('mild', 'moderate', 'severe', 'life_threatening')),
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_dietary_restrictions_user_id ON dietary_restrictions(user_id);
CREATE INDEX idx_dietary_restrictions_item ON dietary_restrictions(item);
```

### Goals Table
```sql
CREATE TABLE user_goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_type VARCHAR(20) CHECK (goal_type IN ('weight_loss', 'weight_gain', 'maintenance', 'muscle_gain', 'health_improvement')),
    current_weight_kg DECIMAL(5,2),
    target_weight_kg DECIMAL(5,2),
    target_bmi DECIMAL(4,2),
    target_date DATE,
    weekly_goal_kg DECIMAL(4,2),
    
    -- แคลอรี่และสารอาหารตามเป้าหมาย (ตาม BMI)
    daily_calorie_target INTEGER, -- คำนวณจาก BMI และเป้าหมาย
    daily_protein_target_g DECIMAL(6,2),
    daily_carb_target_g DECIMAL(6,2),
    daily_fat_target_g DECIMAL(6,2),
    
    -- การติดตามความคืบหน้า
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    last_weight_update TIMESTAMP WITH TIME ZONE,
    
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_user_goals_user_id ON user_goals(user_id);
CREATE INDEX idx_user_goals_status ON user_goals(status);
CREATE INDEX idx_user_goals_target_bmi ON user_goals(target_bmi);
```

### Food Database Table
```sql
CREATE TABLE foods (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name_th VARCHAR(255) NOT NULL,
    name_en VARCHAR(255),
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    brand VARCHAR(100),
    barcode VARCHAR(50),
    
    -- Nutrition per 100g
    calories_per_100g DECIMAL(7,2) NOT NULL,
    protein_per_100g DECIMAL(6,2) DEFAULT 0,
    carbs_per_100g DECIMAL(6,2) DEFAULT 0,
    fat_per_100g DECIMAL(6,2) DEFAULT 0,
    fiber_per_100g DECIMAL(6,2) DEFAULT 0,
    sugar_per_100g DECIMAL(6,2) DEFAULT 0,
    sodium_per_100g DECIMAL(8,2) DEFAULT 0,
    
    -- Serving information
    default_serving_size_g DECIMAL(7,2),
    default_serving_name VARCHAR(50),
    
    -- Metadata
    is_verified BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),
    image_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_foods_name_th ON foods USING gin(to_tsvector('thai', name_th));
CREATE INDEX idx_foods_name_en ON foods USING gin(to_tsvector('english', name_en));
CREATE INDEX idx_foods_category ON foods(category);
CREATE INDEX idx_foods_barcode ON foods(barcode);
```

### Food Ingredients Table
```sql
CREATE TABLE food_ingredients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    food_id UUID NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
    ingredient VARCHAR(100) NOT NULL,
    percentage DECIMAL(5,2),
    is_allergen BOOLEAN DEFAULT FALSE,
    allergen_type VARCHAR(50)
);

CREATE INDEX idx_food_ingredients_food_id ON food_ingredients(food_id);
CREATE INDEX idx_food_ingredients_ingredient ON food_ingredients(ingredient);
CREATE INDEX idx_food_ingredients_allergen ON food_ingredients(is_allergen, allergen_type);
```

### Food Log Entries Table
```sql
CREATE TABLE food_log_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    food_id UUID REFERENCES foods(id),
    custom_food_name VARCHAR(255),
    
    -- Consumption details
    consumed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    meal_type VARCHAR(20) CHECK (meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
    quantity DECIMAL(8,2) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    
    -- Calculated nutrition
    calories DECIMAL(7,2) NOT NULL,
    protein_g DECIMAL(6,2) DEFAULT 0,
    carbs_g DECIMAL(6,2) DEFAULT 0,
    fat_g DECIMAL(6,2) DEFAULT 0,
    fiber_g DECIMAL(6,2) DEFAULT 0,
    
    -- AI Analysis metadata
    image_url VARCHAR(500),
    analysis_id UUID,
    confidence_score DECIMAL(3,2), -- ความแม่นยำจาก AI (0.73 = 73%)
    ai_model_version VARCHAR(20), -- เวอร์ชันโมเดล AI ที่ใช้
    processing_time_ms INTEGER, -- เวลาในการประมวลผล
    
    -- User feedback สำหรับปรับปรุงโมเดล
    user_correction JSONB, -- การแก้ไขจากผู้ใช้
    feedback_accuracy INTEGER CHECK (feedback_accuracy BETWEEN 1 AND 5),
    
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_food_log_user_date ON food_log_entries(user_id, consumed_at);
CREATE INDEX idx_food_log_meal_type ON food_log_entries(meal_type);
CREATE INDEX idx_food_log_analysis_id ON food_log_entries(analysis_id);
CREATE INDEX idx_food_log_confidence ON food_log_entries(confidence_score);
CREATE INDEX idx_food_log_ai_version ON food_log_entries(ai_model_version);
```

### Weight Tracking Table
```sql
CREATE TABLE weight_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    weight_kg DECIMAL(5,2) NOT NULL,
    measured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    measurement_type VARCHAR(20) DEFAULT 'manual' CHECK (measurement_type IN ('manual', 'scale_sync', 'estimated')),
    body_fat_percentage DECIMAL(4,2),
    muscle_mass_kg DECIMAL(5,2),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_weight_entries_user_date ON weight_entries(user_id, measured_at);
```

### Menu Recommendations Table
```sql
CREATE TABLE menu_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_id UUID REFERENCES user_goals(id),
    
    -- Recommendation details
    meal_type VARCHAR(20) CHECK (meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
    recommended_date DATE NOT NULL,
    
    -- Recommended foods
    recommended_foods JSONB NOT NULL, -- Array of food recommendations
    total_calories DECIMAL(7,2),
    total_protein_g DECIMAL(6,2),
    total_carbs_g DECIMAL(6,2),
    total_fat_g DECIMAL(6,2),
    
    -- Recommendation scoring
    health_score DECIMAL(3,2), -- คะแนนความเหมาะสมต่อสุขภาพ
    preference_score DECIMAL(3,2), -- คะแนนความชอบของผู้ใช้
    bmi_compatibility DECIMAL(3,2), -- ความเหมาะสมกับ BMI
    
    -- Algorithm metadata
    algorithm_version VARCHAR(20),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- User interaction
    is_accepted BOOLEAN DEFAULT NULL,
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
    user_feedback TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_menu_recommendations_user_date ON menu_recommendations(user_id, recommended_date);
CREATE INDEX idx_menu_recommendations_meal_type ON menu_recommendations(meal_type);
CREATE INDEX idx_menu_recommendations_health_score ON menu_recommendations(health_score);
```

### AI Model Performance Tracking Table
```sql
CREATE TABLE ai_model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(30) CHECK (model_type IN ('food_detection', 'classification', 'nutrition_estimation')),
    
    -- Performance metrics
    train_accuracy DECIMAL(5,4), -- 0.9800 = 98%
    validation_accuracy DECIMAL(5,4), -- 0.7300 = 73%
    test_accuracy DECIMAL(5,4),
    loss_value DECIMAL(8,6), -- 0.120000
    
    -- Training details
    training_dataset_size INTEGER,
    validation_dataset_size INTEGER,
    training_epochs INTEGER,
    training_duration_hours DECIMAL(6,2),
    
    -- Deployment info
    deployed_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT FALSE,
    performance_notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ai_performance_version ON ai_model_performance(model_version);
CREATE INDEX idx_ai_performance_type ON ai_model_performance(model_type);
CREATE INDEX idx_ai_performance_active ON ai_model_performance(is_active);
```

## MongoDB Collections

### AI Analysis Results
```javascript
// Collection: food_analyses
{
  _id: ObjectId,
  analysisId: "analysis_123",
  userId: "user_456",
  imageUrl: "https://storage.example.com/image.jpg",
  imageMetadata: {
    size: 1024000,
    dimensions: { width: 1920, height: 1080 },
    format: "jpeg"
  },
  processingTime: 2.5,
  modelVersion: "v2.1.0",
  
  // Model performance data
  modelPerformance: {
    trainAccuracy: 0.98, // 98% Train Accuracy
    validationAccuracy: 0.73, // 73% Validation Accuracy
    lossValue: 0.12, // Loss = 0.12
    confidenceThreshold: 0.7
  },
  
  detectedFoods: [
    {
      foodId: "food_789",
      name: "ข้าวผัดกุ้ง",
      boundingBox: { x: 100, y: 150, width: 200, height: 180 },
      confidence: 0.92, // ความแม่นยำการทำนาย
      portionEstimate: {
        amount: 250,
        unit: "grams",
        confidence: 0.85
      },
      nutritionCalculated: {
        calories: 380,
        protein: 18.5,
        carbs: 45.2,
        fat: 12.8
      },
      // BMI compatibility check
      bmiRecommendation: {
        suitable: true,
        userBmi: 23.5,
        recommendation: "Good choice for your BMI range"
      }
    }
  ],
  totalNutrition: {
    calories: 380,
    protein: 18.5,
    carbs: 45.2,
    fat: 12.8
  },
  
  // Health warnings based on user profile
  warnings: [
    {
      type: "allergy",
      severity: "moderate",
      message: "Contains shellfish"
    },
    {
      type: "calorie_excess",
      severity: "low",
      message: "This meal exceeds 30% of your daily calorie target"
    }
  ],
  
  userFeedback: {
    accuracy: 4,
    corrections: [],
    timestamp: ISODate(),
    improvedModel: true // ช่วยปรับปรุงโมเดล
  },
  createdAt: ISODate(),
  updatedAt: ISODate()
}
```

### User Recommendations (แนะนำเมนูอาหารตามเป้าหมายสุขภาพ)
```javascript
// Collection: user_recommendations
{
  _id: ObjectId,
  userId: "user_123",
  date: ISODate("2024-01-01"),
  mealType: "lunch",
  
  // BMI-based recommendations
  userBmiData: {
    currentBmi: 23.5,
    bmiCategory: "normal",
    targetBmi: 22.0,
    dailyCalorieTarget: 1800 // คำนวณจาก BMI
  },
  
  recommendations: [
    {
      foodId: "food_456",
      name: "สลัดไก่ย่าง",
      score: 0.95,
      reasons: [
        "เหมาะสมกับค่า BMI ของคุณ",
        "ช่วยลดน้ำหนักตามเป้าหมาย",
        "โปรตีนสูง คาร์บต่ำ",
        "ไม่เกินแคลอรี่ที่กำหนด"
      ],
      nutritionalFit: {
        calories: 0.9,
        protein: 0.95,
        carbs: 0.8,
        fat: 0.85
      },
      bmiCompatibility: 0.92, // ความเหมาะสมกับ BMI
      healthScore: 0.88,
      userPreferenceScore: 0.88,
      alternatives: ["food_457", "food_458"]
    }
  ],
  
  userContext: {
    currentBmi: 23.5,
    targetBmi: 22.0,
    remainingCalories: 600,
    remainingMacros: {
      protein: 25,
      carbs: 45,
      fat: 15
    },
    recentFoods: ["food_100", "food_101"],
    preferences: ["spicy", "vegetarian"],
    healthGoals: ["weight_loss", "muscle_gain"]
  },
  
  algorithmVersion: "v1.2.0",
  aiModelVersion: "v2.1.0", // เวอร์ชันโมเดล AI ที่ใช้
  createdAt: ISODate(),
  expiresAt: ISODate()
}
```

### User Preferences and Behavior
```javascript
// Collection: user_behavior
{
  _id: ObjectId,
  userId: "user_123",
  preferences: {
    cuisineTypes: ["thai", "japanese", "italian"],
    spiceLevel: "medium",
    cookingMethods: ["grilled", "steamed"],
    avoidedIngredients: ["cilantro", "mushrooms"],
    preferredMealTimes: {
      breakfast: "07:00",
      lunch: "12:00",
      dinner: "19:00"
    }
  },
  behaviorPatterns: {
    avgMealsPerDay: 3.2,
    avgSnacksPerDay: 1.5,
    mostActiveHours: ["07:00-09:00", "12:00-13:00", "18:00-20:00"],
    weekendVsWeekdayDifference: {
      calories: 150,
      mealTiming: 2.5
    }
  },
  foodInteractions: {
    likedFoods: ["food_123", "food_456"],
    dislikedFoods: ["food_789"],
    frequentlyEaten: [
      { foodId: "food_123", frequency: 15, lastEaten: ISODate() }
    ]
  },
  lastUpdated: ISODate()
}
```

### AI Model Training Results
```javascript
// Collection: model_training_results
{
  _id: ObjectId,
  modelId: "kinroo_food_classifier_v2.1.0",
  trainingSession: "training_session_2024_001",
  
  // Training performance metrics
  trainingMetrics: {
    trainAccuracy: 0.98, // 98% ความแม่นยำบน Train Set
    validationAccuracy: 0.73, // 73% ความแม่นยำบน Validation Set
    testAccuracy: 0.75,
    lossValue: 0.12, // ค่า Loss บน Train Set
    validationLoss: 0.28,
    
    // Additional metrics
    precision: 0.74,
    recall: 0.72,
    f1Score: 0.73,
    
    // Per-class performance
    classMetrics: [
      {
        className: "ข้าวผัด",
        accuracy: 0.89,
        precision: 0.87,
        recall: 0.91
      }
    ]
  },
  
  // Training configuration
  trainingConfig: {
    epochs: 150,
    batchSize: 32,
    learningRate: 0.0001,
    optimizer: "Adam",
    lossFunction: "CrossEntropy",
    datasetSize: {
      train: 40000,
      validation: 5000,
      test: 5000
    }
  },
  
  // Model analysis
  modelAnalysis: {
    overfitting: true, // Train Accuracy >> Validation Accuracy
    overfittingScore: 0.25, // (0.98 - 0.73) = 0.25
    recommendations: [
      "เพิ่ม Data Augmentation",
      "ใช้ Regularization techniques",
      "เพิ่มข้อมูล Validation Set",
      "ลด Model Complexity"
    ],
    nextSteps: [
      "Collect more diverse training data",
      "Implement early stopping",
      "Add dropout layers"
    ]
  },
  
  // Deployment info
  deploymentStatus: "production",
  deployedAt: ISODate("2024-01-15"),
  performanceInProduction: {
    averageConfidence: 0.78,
    userSatisfactionScore: 4.2,
    correctionRate: 0.15 // 15% ของการทำนายถูกแก้ไขโดยผู้ใช้
  },
  
  createdAt: ISODate(),
  updatedAt: ISODate()
}
```

## Redis Cache Structure

### Session Management
```
Key: session:{userId}
Value: {
  "token": "jwt_token",
  "refreshToken": "refresh_token",
  "expiresAt": 1640995200,
  "deviceInfo": {...}
}
TTL: 7 days
```

### Daily Nutrition Cache (รวม BMI-based targets)
```
Key: nutrition:daily:{userId}:{date}
Value: {
  "consumed": {
    "calories": 1850,
    "protein": 95,
    "carbs": 180,
    "fat": 65
  },
  "target": {
    "calories": 1800, // คำนวณจาก BMI และเป้าหมาย
    "protein": 135,
    "carbs": 225,
    "fat": 60
  },
  "bmiData": {
    "currentBmi": 23.5,
    "targetBmi": 22.0,
    "bmiCategory": "normal",
    "recommendedCalories": 1800
  },
  "remaining": {
    "calories": -50, // เกินเป้าหมาย
    "protein": 40,
    "carbs": 45,
    "fat": -5
  },
  "aiRecommendations": [
    "ลดแคลอรี่ 50 แคลอรี่ในมื้อเย็น",
    "เพิ่มผักใบเขียว",
    "ลดไขมันอิ่มตัว"
  ],
  "lastUpdated": "2024-01-01T15:30:00Z"
}
TTL: 24 hours
```

### Food Search Cache
```
Key: food:search:{query_hash}
Value: [
  {
    "id": "food_123",
    "name": "ข้าวผัด",
    "calories": 200,
    "category": "main_dish"
  }
]
TTL: 1 hour
```

## InfluxDB Time Series Schema

### Nutrition Metrics
```
Measurement: nutrition_daily
Tags:
  - user_id
  - goal_id
Fields:
  - calories_consumed (float)
  - calories_target (float)
  - protein_g (float)
  - carbs_g (float)
  - fat_g (float)
  - water_ml (float)
Time: timestamp
```

### Weight Tracking และ BMI Monitoring
```
Measurement: weight_tracking
Tags:
  - user_id
  - measurement_type
  - bmi_category
Fields:
  - weight_kg (float)
  - height_cm (float)
  - bmi (float)
  - bmi_change (float) // การเปลี่ยนแปลง BMI
  - body_fat_percentage (float)
  - muscle_mass_kg (float)
  - target_weight_kg (float)
  - weight_goal_progress (float) // ความคืบหน้าเป้าหมาย
Time: timestamp
```

### AI Model Performance Monitoring
```
Measurement: ai_model_performance
Tags:
  - model_version
  - model_type
  - deployment_environment
Fields:
  - train_accuracy (float) // 0.98
  - validation_accuracy (float) // 0.73
  - loss_value (float) // 0.12
  - prediction_confidence (float)
  - user_correction_rate (float)
  - processing_time_ms (integer)
  - daily_predictions (integer)
Time: timestamp
```

### App Usage Analytics
```
Measurement: app_usage
Tags:
  - user_id
  - feature
  - platform
Fields:
  - session_duration (integer)
  - actions_count (integer)
  - photos_analyzed (integer)
Time: timestamp
```

## Database Optimization Strategies

### Indexing Strategy
- Composite indexes for common query patterns
- Partial indexes for filtered queries
- GIN indexes for full-text search
- Time-based partitioning for large tables

### Partitioning
```sql
-- Partition food_log_entries by month
CREATE TABLE food_log_entries_y2024m01 PARTITION OF food_log_entries
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Data Retention Policies
- Food logs: Keep 2 years, archive older data
- AI analysis results: Keep 6 months
- Cache data: TTL-based expiration
- Analytics data: Aggregate and downsample after 1 year

### Backup Strategy
- PostgreSQL: Daily full backups, continuous WAL archiving
- MongoDB: Replica sets with automated backups
- Redis: RDB snapshots every 6 hours
- InfluxDB: Incremental backups with retention policies