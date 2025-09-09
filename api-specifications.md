# กินรู้ API Specifications

## API Overview
RESTful API design with GraphQL for complex queries, following OpenAPI 3.0 specifications.

## Authentication Endpoints

### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "profile": {
    "firstName": "John",
    "lastName": "Doe",
    "dateOfBirth": "1990-01-01",
    "gender": "male",
    "height": 175,
    "weight": 70,
    "activityLevel": "moderate",
    "dietaryRestrictions": ["vegetarian"],
    "allergies": ["nuts", "dairy"],
    "healthConditions": ["diabetes"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user_123",
      "email": "user@example.com",
      "profile": { ... },
      "createdAt": "2024-01-01T00:00:00Z"
    },
    "token": "jwt_token_here",
    "refreshToken": "refresh_token_here"
  }
}
```

### POST /auth/login
Authenticate user and return access token.

### POST /auth/refresh
Refresh expired access token.

### POST /auth/logout
Invalidate user session.

## Food Analysis Endpoints

### POST /food/analyze
Analyze food image and return nutritional information.

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "mealType": "lunch",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysisId": "analysis_123",
    "foods": [
      {
        "id": "food_456",
        "name": "ข้าวผัดกุ้ง",
        "nameEn": "Shrimp Fried Rice",
        "category": "main_dish",
        "portion": {
          "amount": 250,
          "unit": "grams"
        },
        "nutrition": {
          "calories": 380,
          "protein": 18.5,
          "carbohydrates": 45.2,
          "fat": 12.8,
          "fiber": 2.1,
          "sugar": 3.4,
          "sodium": 890
        },
        "confidence": 0.92,
        "boundingBox": {
          "x": 100,
          "y": 150,
          "width": 200,
          "height": 180
        }
      }
    ],
    "totalNutrition": {
      "calories": 380,
      "protein": 18.5,
      "carbohydrates": 45.2,
      "fat": 12.8
    },
    "warnings": [
      {
        "type": "allergy",
        "message": "This dish may contain shellfish"
      }
    ]
  }
}
```

### GET /food/search
Search food database by name or barcode.

**Query Parameters:**
- `q`: Search query
- `barcode`: Product barcode
- `category`: Food category filter
- `limit`: Results limit (default: 20)

### GET /food/{foodId}
Get detailed information about a specific food item.

## User Profile Endpoints

### GET /user/profile
Get current user profile and preferences.

### PUT /user/profile
Update user profile information.

### POST /user/goals
Create or update user health goals.

**Request Body:**
```json
{
  "type": "weight_loss",
  "targetWeight": 65,
  "targetDate": "2024-06-01",
  "weeklyGoal": 0.5,
  "calorieDeficit": 500
}
```

### GET /user/daily-needs
Calculate daily nutritional requirements based on user profile.

**Response:**
```json
{
  "success": true,
  "data": {
    "calories": 2200,
    "protein": 137.5,
    "carbohydrates": 247.5,
    "fat": 73.3,
    "water": 2500,
    "fiber": 25
  }
}
```

## Food Logging Endpoints

### POST /food-log/entry
Log a food consumption entry.

**Request Body:**
```json
{
  "date": "2024-01-01",
  "mealType": "breakfast",
  "foods": [
    {
      "foodId": "food_123",
      "quantity": 1,
      "unit": "serving",
      "customNutrition": {
        "calories": 250
      }
    }
  ],
  "imageUrl": "https://storage.example.com/image.jpg",
  "notes": "Homemade with less oil"
}
```

### GET /food-log/daily
Get daily food log for a specific date.

**Query Parameters:**
- `date`: Date in YYYY-MM-DD format
- `timezone`: User timezone

### GET /food-log/range
Get food log data for a date range.

**Query Parameters:**
- `startDate`: Start date
- `endDate`: End date
- `groupBy`: Group by day/week/month

### PUT /food-log/entry/{entryId}
Update existing food log entry.

### DELETE /food-log/entry/{entryId}
Delete food log entry.

## Recommendations Endpoints

### GET /recommendations/foods
Get personalized food recommendations.

**Query Parameters:**
- `mealType`: breakfast/lunch/dinner/snack
- `remainingCalories`: Remaining daily calories
- `preferences`: User preferences
- `limit`: Number of recommendations

**Response:**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "food": {
          "id": "food_789",
          "name": "สลัดผักรวม",
          "nutrition": { ... }
        },
        "score": 0.95,
        "reason": "High in fiber, fits your remaining calorie budget",
        "alternatives": ["food_790", "food_791"]
      }
    ],
    "totalRecommendations": 15
  }
}
```

### GET /recommendations/meals
Get complete meal recommendations.

### POST /recommendations/feedback
Provide feedback on recommendations to improve algorithm.

## Analytics Endpoints

### GET /analytics/dashboard
Get dashboard analytics data.

**Response:**
```json
{
  "success": true,
  "data": {
    "today": {
      "caloriesConsumed": 1850,
      "caloriesRemaining": 350,
      "macros": {
        "protein": { "consumed": 95, "target": 137.5 },
        "carbs": { "consumed": 180, "target": 247.5 },
        "fat": { "consumed": 65, "target": 73.3 }
      },
      "waterIntake": 1800,
      "exerciseCalories": 300
    },
    "weeklyAverage": {
      "calories": 2100,
      "protein": 120,
      "carbs": 220,
      "fat": 70
    },
    "streaks": {
      "logging": 15,
      "goalMet": 8
    }
  }
}
```

### GET /analytics/trends
Get nutrition and weight trends over time.

### GET /analytics/reports
Generate detailed nutrition reports.

## Progress Tracking Endpoints

### POST /progress/weight
Log weight measurement.

**Request Body:**
```json
{
  "weight": 68.5,
  "date": "2024-01-01",
  "notes": "Morning weight after workout"
}
```

### GET /progress/weight
Get weight history and trends.

### GET /progress/goals
Get progress towards current goals.

### POST /progress/photos
Upload progress photos.

## Notification Endpoints

### GET /notifications
Get user notifications.

### PUT /notifications/{notificationId}/read
Mark notification as read.

### POST /notifications/preferences
Update notification preferences.

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

## Rate Limiting

- Authentication endpoints: 5 requests per minute
- Food analysis: 10 requests per minute
- General API: 100 requests per minute
- Premium users: 2x rate limits

## Webhooks

### POST /webhooks/analysis-complete
Triggered when food analysis is completed (for async processing).

### POST /webhooks/goal-achieved
Triggered when user achieves a milestone.

## API Versioning

All endpoints are versioned using URL path versioning:
- Current version: `/api/v1/`
- Beta features: `/api/v2/`

## Authentication

All protected endpoints require Bearer token authentication:
```
Authorization: Bearer <jwt_token>
```

## Data Formats

- Dates: ISO 8601 format (YYYY-MM-DDTHH:mm:ssZ)
- Numbers: Decimal format with appropriate precision
- Images: Base64 encoded or multipart/form-data
- Weights: Kilograms (kg)
- Heights: Centimeters (cm)
- Calories: kcal
- Macronutrients: grams (g)