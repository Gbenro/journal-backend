# Journal Backend Service

FastAPI backend service for the GPT Agent Journaling System. This service provides the database layer and core CRUD operations for journal entries.

## 🚀 Railway Deployment

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new)

### Quick Deploy
1. Click the Railway button above or go to [Railway](https://railway.app)
2. Connect this GitHub repository
3. Add a PostgreSQL database to your project
4. Set the `DATABASE_URL` environment variable (Railway auto-configures this)
5. Deploy!

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (auto-configured by Railway)
- `PORT`: Service port (auto-configured by Railway)

## 🏗️ Architecture

This backend service:
- Handles database operations for journal entries
- Provides RESTful API endpoints
- Manages PostgreSQL connections with Railway optimization
- Includes health monitoring

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

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost/journal"

# Run the service
uvicorn main:app --reload --port 8000
```

## 🔗 Related Services

- **Middleware Service**: [journal-middleware-repo](../journal-middleware-repo) - API gateway with authentication
- **Main Project**: [Scribe_agent](../) - Complete journaling system documentation

## 📦 Dependencies

- FastAPI - Web framework
- SQLAlchemy - ORM and database toolkit
- PostgreSQL - Database (provided by Railway)
- Uvicorn - ASGI server

## 🏷️ Version

1.0.0