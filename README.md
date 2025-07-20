# Mirror Scribe Backend with Persistent Storage

FastAPI backend service for the GPT Agent Journaling System with intelligent tagging and persistent storage. This service provides the database layer and core CRUD operations using SQLite with Railway persistent volumes for maximum reliability and data persistence.

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
- Uses SQLite database with persistent volume storage
- Handles database operations for journal entries with intelligent tagging
- Provides RESTful API endpoints with auto-tagging capabilities
- Includes comprehensive health monitoring and storage statistics
- Data persists across Railway deployments and container restarts
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
# Health check with storage info
curl https://your-backend.railway.app/health

# Storage configuration details
curl https://your-backend.railway.app/storage-info

# Save a message with tags
curl -X POST https://your-backend.railway.app/api/save \
  -H "Content-Type: application/json" \
  -d '{"content": "Fixed deployment bug today!", "user_id": "user123", "manual_tags": ["work"], "auto_tag": true}'

# Get messages with tag filtering
curl https://your-backend.railway.app/api/messages/user123?tags=work,coding

# Get all available tags
curl https://your-backend.railway.app/api/tags

# Get tag suggestions
curl -X POST https://your-backend.railway.app/api/tags/suggestions \
  -H "Content-Type: application/json" \
  -d '{"content": "Had dinner with family", "limit": 5}'

# Get comprehensive statistics
curl https://your-backend.railway.app/stats
```

## 💾 Persistent Storage

The backend uses Railway's persistent volumes to ensure data survives across deployments:

- **Local Development**: Database stored as `journal.db` in current directory
- **Railway Production**: Database stored at `/app/data/journal.db` on persistent volume
- **Automatic Detection**: Environment-aware path configuration
- **Zero Data Loss**: All journal entries and tags persist through redeploys

## 🔗 Related Services

- **Middleware Service**: [journal-middleware-repo](../journal-middleware-repo) - API gateway with authentication
- **Main Project**: [Scribe_agent](../) - Complete journaling system documentation

## 📦 Dependencies

- FastAPI - Web framework
- SQLite - Built-in database (no setup required)
- Uvicorn - ASGI server

## 🏷️ Version

1.0.0