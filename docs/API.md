# Video Searcher API Documentation

## Overview
This document describes the API endpoints and usage for the Video Searcher system.

### Authentication
All API endpoints require authentication using an API key passed in the header:
```
Authorization: Bearer <api_key>
```

### Endpoints

#### Process Video
POST /api/v1/process
- Request body: multipart/form-data with video file
- Response: JSON with processing status and metadata

#### Search Videos  
GET /api/v1/search
- Query parameters: q (search query)
- Response: JSON array of matching videos