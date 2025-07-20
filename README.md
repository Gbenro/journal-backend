# Journal Backend Service

FastAPI backend service for the GPT Agent Journaling System. This service provides the database layer and core CRUD operations for journal entries using SQLite for maximum reliability.

## 🚀 Railway Deployment

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new)

### Quick Deploy
1. Click the Railway button above or go to [Railway](https://railway.app)
2. Connect this GitHub repository
3. Deploy! (No database setup required - uses built-in SQLite)

### Environment Variables
- `PORT`: Service port (auto-configured by Railway)

## 🏗️ Architecture

This backend service:
- Uses SQLite database for zero-config reliability
- Handles database operations for journal entries
- Provides RESTful API endpoints
- Includes health monitoring and statistics
- No external database dependencies

## 📚 API Endpoints

### Health Check
```
GET /health
```
Returns service health and database connectivity status.

### Save Entry
```
POST /api/save
Content-Type: application/json

{
  "content": "Journal entry content",
  "user_id": "user123"
}
```

### Get User Entries
```
GET /api/messages/{user_id}?limit=100&offset=0
```
Returns paginated journal entries for a user.

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service (SQLite database auto-created)
uvicorn main:app --reload --port 8000
```

## 🧪 Testing Endpoints

```bash
# Health check
curl https://your-backend.railway.app/health

# Save a message
curl -X POST https://your-backend.railway.app/api/save \
  -H "Content-Type: application/json" \
  -d '{"content": "My first journal entry", "user_id": "user123"}'

# Get messages
curl https://your-backend.railway.app/api/messages/user123

# Get statistics
curl https://your-backend.railway.app/stats
```

## 🔗 Related Services

- **Middleware Service**: [journal-middleware-repo](../journal-middleware-repo) - API gateway with authentication
- **Main Project**: [Scribe_agent](../) - Complete journaling system documentation

## 📦 Dependencies

- FastAPI - Web framework
- SQLite - Built-in database (no setup required)
- Uvicorn - ASGI server

## 🏷️ Version

1.0.0